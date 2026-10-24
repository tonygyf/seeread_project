"""Deterministic feature-grid TSR baseline for V5 detection.

Assumptions are explicit: standard FPS, a fixed rectangle width in FEATURE
pixels, and optional morphological closing. These are reimplementation choices
until the author's exact decoder and thresholds are available. Geometry never
uses labels/SCM. Polygon vertices are scaled x/y independently on output.
"""
from __future__ import annotations
import cv2
import numpy as np


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def _single(value, channels):
    value = _array(value)
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError('decode accepts one image; index each batch item explicitly')
        value = value[0]
    if value.ndim != 3 or value.shape[0] != channels:
        raise ValueError(f'Expected ({channels},H,W) or (1,{channels},H,W), got {value.shape}')
    return value


def farthest_point_sampling(points, count):
    """Stable row-order tie breaking; no RNG or data-dependent seed."""
    count = min(int(count), len(points))
    if count <= 0:
        return np.empty(0, np.int64)
    first = int(np.argmax(((points - points.mean(0)) ** 2).sum(1)))
    selected = [first]
    distance = ((points - points[first]) ** 2).sum(1)
    for _ in range(count - 1):
        index = int(np.argmax(distance))
        selected.append(index)
        distance = np.minimum(distance, ((points - points[index]) ** 2).sum(1))
    return np.asarray(selected, np.int64)


def _positive_probability(logits):
    logits = logits - logits.max(0, keepdims=True)
    exp = np.exp(logits)
    return exp[1] / exp.sum(0)


def decode(pred, stride=None, image_hw=None, text_threshold=.5, center_threshold=.5,
           rectangle_width=6., sample_step=3., max_samples=64, close_kernel=3,
           min_component_area=3, min_polygon_area=3, max_detections=100):
    """One image -> list of {'poly': Nx2 image-pixel ndarray, 'score': float}.

    Specify exactly one of image_hw=(H,W) or stride=scalar/(stride_y,stride_x).
    Both text/center inputs are two-class logits, height full feature-pixel units,
    angle tangent radians. There is no implicit height exp/log transform.
    """
    if (stride is None) == (image_hw is None):
        raise ValueError('Specify exactly one of stride or image_hw')
    if rectangle_width <= 0 or sample_step <= 0 or max_samples < 1:
        raise ValueError('Positive rectangle width, sample step and max_samples are required')
    if close_kernel < 0 or (close_kernel and close_kernel % 2 != 1):
        raise ValueError('close_kernel must be zero or a positive odd integer')
    if not 0 <= text_threshold <= 1 or not 0 <= center_threshold <= 1:
        raise ValueError('Probability thresholds must lie in [0,1]')
    text, center = _single(pred['text'], 2), _single(pred['center'], 2)
    height, angle = _single(pred['height'], 1)[0], _single(pred['angle'], 1)[0]
    if text.shape[1:] != center.shape[1:] or text.shape[1:] != height.shape or height.shape != angle.shape:
        raise ValueError('Prediction maps must share spatial dimensions')
    if not all(np.isfinite(x).all() for x in (text, center, height, angle)):
        raise ValueError('Nonfinite detector output')
    fh, fw = height.shape
    if image_hw is not None:
        scale = np.array([image_hw[1] / fw, image_hw[0] / fh], np.float32)
    else:
        sy, sx = (float(stride), float(stride)) if np.isscalar(stride) else map(float, stride)
        scale = np.array([sx, sy], np.float32)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Image scale must be positive and finite')
    tp, cp = _positive_probability(text), _positive_probability(center)
    mask = ((tp >= text_threshold) & (cp >= center_threshold)).astype(np.uint8)
    n, labels = cv2.connectedComponents(mask, connectivity=8)
    results = []
    for component in range(1, n):
        yy, xx = np.nonzero(labels == component)
        if len(xx) < min_component_area:
            continue
        points = np.stack([xx, yy], 1).astype(np.float32)
        extent = np.linalg.norm(np.ptp(points, axis=0))
        count = min(max_samples, max(1, int(np.ceil(extent / sample_step)) + 1))
        selected = points[farthest_point_sampling(points, count)]
        accumulated = np.zeros((fh, fw), np.uint8)
        for x, y in selected:
            ix, iy = int(x), int(y)
            h, a = max(float(height[iy, ix]), 0.), float(angle[iy, ix])
            if h <= 0:
                continue
            tangent = np.array([np.cos(a), np.sin(a)], np.float32)
            normal = np.array([-tangent[1], tangent[0]], np.float32)
            c = np.array([x, y], np.float32)
            along, across = tangent * rectangle_width / 2, normal * h / 2
            quad = np.stack([c - along - across, c + along - across,
                             c + along + across, c - along + across])
            cv2.fillConvexPoly(accumulated, np.rint(quad).astype(np.int32), 1)
        if close_kernel:
            accumulated = cv2.morphologyEx(accumulated, cv2.MORPH_CLOSE,
                                            np.ones((close_kernel, close_kernel), np.uint8))
        contours, _ = cv2.findContours(accumulated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < min_polygon_area:
            continue
        polygon = cv2.approxPolyDP(contour, .005 * cv2.arcLength(contour, True), True).reshape(-1, 2).astype(np.float32)
        if len(polygon) < 3:
            continue
        interior = np.zeros((fh, fw), np.uint8)
        cv2.fillPoly(interior, [contour], 1)
        results.append(dict(poly=polygon * scale, score=float(tp[interior > 0].mean())))
    results.sort(key=lambda item: -item['score'])
    return results[:max_detections] if max_detections else results
