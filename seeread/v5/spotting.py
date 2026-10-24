"""Unbenchmarked V5 composition: Seeing polygons -> native crop -> Reading.

This inference adapter never loads legacy checkpoints, runs TSR centerlines, or
uses split_polygon/minAreaRect. Geometry assumes one PCA cross-section per word
(no principal-axis fold). Reading direction is canonical +X for horizontal and
+Y for vertical words; polygons alone cannot determine 180-degree text direction
or arbitrary vertical-language order. These are explicit baseline limitations.
"""
from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from shapely.geometry import Polygon

from seeread.models.rectify import rectify_strip  # sampling function only
from seeread.v5.recognition import ReadingPipeline
from seeread.v5.seeing import SeeingDetector
from seeread.v5.seeing_data import _cross_sections, _principal_axis
from seeread.v5.seeing_decode import decode


@dataclass
class CropResult:
    crop: torch.Tensor | None
    reason: str | None
    centers: np.ndarray | None = None
    heights: np.ndarray | None = None
    sampling: str | None = None
    fallback_reason: str | None = None


def _image(image):
    if not isinstance(image, torch.Tensor):
        raise TypeError('Expected RGB torch tensor 3,H,W or 1,3,H,W')
    if image.ndim == 3:
        image = image[None]
    if image.ndim != 4 or image.shape[:2] != (1, 3) or min(image.shape[-2:]) < 2:
        raise ValueError('Expected one RGB image with spatial dimensions >= 2')
    if image.dtype == torch.uint8:
        image = image.float() / 255
    elif image.is_floating_point():
        image = image.float()
    else:
        raise TypeError('RGB tensor must be floating [0,1] or uint8 [0,255]')
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError('RGB image must be finite and in [0,1]')
    return image


def _area_axis(polygon):
    # Exact polygon area moments avoid bias from uneven annotation vertices
    # and tiny endpoint tilts introduced by finite perimeter sampling.
    moments = cv2.moments(polygon)
    covariance = np.array([[moments['mu20'], moments['mu11']],
                           [moments['mu11'], moments['mu02']]]) / moments['m00']
    _, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, -1].astype(np.float32)
    return _principal_axis(np.stack([axis, -axis])).astype(np.float32)


def _single_sections(polygon, axis, locations):
    u = polygon @ axis
    end = np.roll(u, -1)
    span = np.ptp(u)
    q = np.clip(locations, u.min() + 1e-5 * span, u.max() - 1e-5 * span)[:, None]
    active = (abs(end - u) > 1e-7)[None] & (q >= np.minimum(u, end)[None]) & (q < np.maximum(u, end)[None])
    return active.sum(1)


def _rectify_boundaries(image, centers, top, bottom, output_hw, pad):
    """Interpolate actual polygon boundaries, retaining every boundary breakpoint."""
    centers, top, bottom = [torch.as_tensor(value, dtype=image.dtype, device=image.device)
                            for value in (centers, top, bottom)]
    top = centers + (top - centers) * pad
    bottom = centers + (bottom - centers) * pad
    distance = (centers[1:] - centers[:-1]).norm(dim=1)
    arc = torch.cat([distance.new_zeros(1), distance.cumsum(0)])
    arc = arc / arc[-1]
    columns = torch.linspace(0, 1, output_hw[1], device=image.device)
    index = torch.searchsorted(arc.contiguous(), columns, right=True).clamp(1, len(arc) - 1)
    weight = ((columns - arc[index - 1]) /
              (arc[index] - arc[index - 1]).clamp_min(1e-8)).clamp(0, 1)[:, None]
    top = top[index - 1] * (1 - weight) + top[index] * weight
    bottom = bottom[index - 1] * (1 - weight) + bottom[index] * weight
    rows = torch.linspace(0, 1, output_hw[0], device=image.device)[:, None, None]
    pixels = top[None] * (1 - rows) + bottom[None] * rows
    height, width = image.shape[-2:]
    grid = torch.stack([pixels[..., 0] / (width - 1) * 2 - 1,
                        pixels[..., 1] / (height - 1) * 2 - 1], -1)[None]
    return F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)


def _polygon_crop(image, polygon, output_hw=(384, 384), center_points=32, pad=1., min_mean_height=1.,
                  sampling='local_normal'):
    if sampling == 'boundary_when_supported':
        result = _polygon_crop(image, polygon, output_hw, center_points, pad, min_mean_height, 'polygon_boundary')
        if result.reason == 'multiple_cross_sections':
            result = _polygon_crop(image, polygon, output_hw, center_points, pad, min_mean_height, 'local_normal')
            result.sampling = 'local_normal'
            result.fallback_reason = 'multiple_cross_sections'
        return result
    if sampling not in ('local_normal', 'polygon_boundary'):
        raise ValueError('Crop sampling must be local_normal or polygon_boundary')
    if len(output_hw) != 2 or min(output_hw) <= 0 or center_points < 2 or not np.isfinite(pad) or pad <= 0:
        raise ValueError('Positive crop dimensions/pad and center_points >= 2 are required')
    try:
        polygon = np.asarray(polygon, np.float32)
    except (ValueError, TypeError):
        return CropResult(None, 'polygon_not_numeric')
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        return CropResult(None, 'polygon_too_few_vertices')
    if not np.isfinite(polygon).all():
        return CropResult(None, 'polygon_not_finite')
    if not Polygon(polygon).is_valid:
        return CropResult(None, 'polygon_invalid_topology')
    if abs(cv2.contourArea(polygon)) < 1e-3:
        return CropResult(None, 'polygon_area_too_small')
    h, w = image.shape[-2:]
    if (polygon < 0).any() or (polygon[:, 0] > w - 1).any() or (polygon[:, 1] > h - 1).any():
        return CropResult(None, 'polygon_outside_image')
    axis = _area_axis(polygon)
    normal = np.array([-axis[1], axis[0]], np.float32)
    projected = polygon @ axis
    if np.ptp(projected) < 1e-4:
        return CropResult(None, 'centerline_degenerate')
    locations = np.linspace(float(projected.min()), float(projected.max()), int(center_points)).astype(np.float32)
    if sampling == 'polygon_boundary':
        # Boundary slopes change at projected vertices. Omitting those knots
        # can interpolate across a concavity and erase part of a character.
        locations = np.unique(np.clip(np.r_[locations, projected],
                                      projected.min() + 1e-5 * np.ptp(projected),
                                      projected.max() - 1e-5 * np.ptp(projected)))
    counts = _single_sections(polygon, axis, locations)
    if (counts > 2).any():
        return CropResult(None, 'multiple_cross_sections')
    if (counts != 2).any():
        return CropResult(None, 'missing_cross_section')
    origin_v = float(polygon.mean(0) @ normal)
    query = locations[:, None] * axis[None] + origin_v * normal[None]
    midpoint, heights, _, valid, cross_height = _cross_sections(polygon, query, axis)
    centers = locations[:, None] * axis[None] + midpoint[:, None] * normal[None]
    if not valid.all() or not np.isfinite(centers).all() or not np.isfinite(heights).all():
        return CropResult(None, 'invalid_cross_section')
    if heights.mean() < min_mean_height or (heights <= 0).any():
        return CropResult(None, 'height_too_small')
    if np.linalg.norm(np.diff(centers, axis=0), axis=1).sum() < 1e-3:
        return CropResult(None, 'centerline_degenerate')
    if sampling == 'polygon_boundary':
        top = centers - cross_height[:, None] * normal[None] / 2
        bottom = centers + cross_height[:, None] * normal[None] / 2
        crop = _rectify_boundaries(image, centers, top, bottom, tuple(map(int, output_hw)), float(pad))
    else:
        crop = rectify_strip(image, torch.as_tensor(centers, dtype=image.dtype, device=image.device),
                             torch.as_tensor(heights, dtype=image.dtype, device=image.device),
                             out_hw=tuple(map(int, output_hw)), pad=float(pad))
    if not torch.isfinite(crop).all():
        return CropResult(None, 'crop_not_finite', centers, heights)
    # Bilinear weights can sum to 1+float32 epsilon on saturated pixels.
    # Preserve the RGB input contract before the unchanged Reading enhancer.
    return CropResult(crop.clamp(0., 1.), None, centers, heights, sampling=sampling)


def polygon_crop(image, polygon, output_hw=(384, 384), center_points=32, pad=1., min_mean_height=1.,
                 sampling='local_normal'):
    """One original-coordinate polygon -> crop, with explicit invalid reason.

    This exact geometry is shared by predicted-polygon and GT-crop diagnostics.
    The output goes directly to Reading at 384x384 by default. pad=1.0 samples
    the polygon-derived strip; any added padding is an explicit later ablation.
    polygon_boundary samples the actual cross-section edges; local_normal
    preserves the original symmetric centerline strip for controlled comparison.
    Outside-image polygons are reported invalid rather than silently clipped.
    """
    return _polygon_crop(_image(image), polygon, output_hw, center_points, pad, min_mean_height, sampling)


def crop_exposure(crop, mode='none'):
    """Image-only connector ablation: bounded RGB gain before the fixed enhancer."""
    if mode == 'none':
        return crop, 1.
    if mode != 'p95':
        raise ValueError('Crop exposure must be none or p95')
    luminance = (crop * crop.new_tensor([.2126, .7152, .0722])[None, :, None, None]).sum(1)
    p95 = torch.quantile(luminance.float(), .95)
    gain = float((.7 / p95.clamp_min(1e-6)).clamp(1., 8.))
    return (crop * gain).clamp(0., 1.), gain


def _device(module, fallback):
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    return buffer.device if buffer is not None else fallback


class V5Spotter(nn.Module):
    """V5-only composition; supports any real Reading enhancer via its pipeline."""
    def __init__(self, detector, reading, detector_hw=(640, 640), rec_chunk=8,
                 decode_kwargs=None, crop_kwargs=None, exposure='none'):
        super().__init__()
        if not isinstance(detector, SeeingDetector) or not isinstance(reading, ReadingPipeline):
            raise TypeError('V5Spotter requires V5 SeeingDetector and ReadingPipeline components')
        if not isinstance(rec_chunk, int) or isinstance(rec_chunk, bool) or not 1 <= rec_chunk <= 8:
            raise ValueError('Recognition chunk must be an integer in [1,8]')
        if len(detector_hw) != 2 or min(detector_hw) <= 0:
            raise ValueError('detector_hw must contain positive H,W')
        self.detector, self.reading = detector, reading
        self.detector_hw, self.rec_chunk = tuple(map(int, detector_hw)), rec_chunk
        self.decode_kwargs, self.crop_kwargs = dict(decode_kwargs or {}), dict(crop_kwargs or {})
        if exposure not in ('none', 'p95'):
            raise ValueError('Crop exposure must be none or p95')
        self.exposure = exposure
        if {'stride', 'image_hw'} & self.decode_kwargs.keys():
            raise ValueError('V5Spotter owns inverse geometry; do not override decoder stride/image_hw')
        self.provenance = dict(status='composition implemented; no benchmark claim', detector_hw=list(self.detector_hw),
                               recognition_hw=list(self.crop_kwargs.get('output_hw', (384, 384))), rec_chunk=rec_chunk,
                               crop={**dict(center_points=32, pad=1., min_mean_height=1., sampling='local_normal'), **self.crop_kwargs},
                               exposure=dict(mode=exposure, target_p95=.7, maximum_gain=8.),
                               assumptions=['PCA cross-sections; conditional strategy retries only multiple_cross_sections with local normals',
                                            'canonical +X/+Y reading order; no direction oracle',
                                            'sampling policy is recorded in crop and each instance crop_strategy',
                                            'invalid crop retains detection with empty text and reason'],
                               detector=getattr(detector, 'provenance', {}),
                               recognizer=getattr(reading.recognizer, 'provenance', {}),
                               enhancer=getattr(reading.enhancer, 'provenance', {}))

    def _recognize(self, image, detections):
        output = [dict(detection, instance_id=index, text='', crop_error=None)
                  for index, detection in enumerate(detections)]
        pending, indices = [], []
        device = _device(self.reading, image.device)

        def flush():
            if not pending:
                return
            predictions = self.reading.predict(torch.cat(pending).to(device))
            if len(predictions) != len(indices):
                raise ValueError('Recognizer returned the wrong number of predictions')
            for index, prediction in zip(indices, predictions):
                if not isinstance(prediction, str):
                    raise TypeError('Recognizer returned a non-string prediction')
                output[index]['text'] = prediction
            pending.clear()
            indices.clear()

        for index, detection in enumerate(output):
            result = _polygon_crop(image, detection['poly'], **self.crop_kwargs)
            detection['crop_strategy'] = result.sampling or self.crop_kwargs.get('sampling', 'local_normal')
            detection['crop_fallback_reason'] = result.fallback_reason
            detection['crop_exposure_gain'] = None
            if result.reason is not None:
                detection['crop_error'] = result.reason
                continue
            crop, gain = crop_exposure(result.crop, self.exposure)
            detection['crop_exposure_gain'] = gain
            pending.append(crop)
            indices.append(index)
            if len(pending) == self.rec_chunk:
                flush()
        flush()
        return output

    @torch.no_grad()
    def predict(self, image):
        self.eval()
        image = _image(image)
        detector_image = F.interpolate(image, size=self.detector_hw, mode='bilinear', align_corners=False)
        detector_image = detector_image.to(_device(self.detector, image.device))
        # Image-only call: GT masks and SCM are never part of inference.
        pred = self.detector(detector_image)
        detections = decode(pred, image_hw=tuple(image.shape[-2:]), **self.decode_kwargs)
        return self._recognize(image, detections)

    @torch.no_grad()
    def recognize_polygons(self, image, polygons):
        """GT/fixed-polygon diagnostic; detector is never invoked."""
        self.reading.eval()
        image = _image(image)
        return self._recognize(image, [dict(poly=np.asarray(poly, np.float32), score=1.) for poly in polygons])

    def forward(self, image):
        return self.predict(image)
