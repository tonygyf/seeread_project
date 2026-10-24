"""Polygon-native detection targets for the explicitly assumed V5 reimplementation.

Coordinates are scaled independently in x/y into the feature grid. Full height
and tangent angle are measured THERE, before any inverse image scaling.

Geometry assumption: PCA cross sections describe each nonfolding text branch.
Intersections use the actual polygon boundary (any vertex count/orientation),
not a minimum-area rectangle. Multiple cross-section intervals are kept separate
and each raster pixel uses its own containing/nearest interval. At a split or a
principal-axis fold the tangent is not uniquely defined; this is a documented
approximation, not an assertion of the unavailable author's target generator.
"""
from __future__ import annotations
import cv2
import numpy as np


def _principal_axis(points):
    if len(points) < 2:
        return np.array([1., 0.], np.float32)
    axis = np.linalg.svd(points - points.mean(0), full_matrices=False)[2][0]
    if (abs(axis[0]) >= abs(axis[1]) and axis[0] < 0) or (abs(axis[0]) < abs(axis[1]) and axis[1] < 0):
        axis = -axis
    return axis


def _cross_sections(poly, pixels, axis):
    """Return midpoint, full orthogonal height and tangent at every text pixel."""
    normal = np.array([-axis[1], axis[0]])
    u, v = pixels @ axis, pixels @ normal
    pu, pv = poly @ axis, poly @ normal
    span = float(np.ptp(pu))
    if span < 1e-6:
        z = np.zeros(len(pixels), np.float32)
        return z, z, z, np.zeros(len(pixels), bool), z
    # Half-open edge membership avoids counting a shared vertex twice. Evaluate
    # extremal sections just inside the polygon, where endpoints have a limit.
    query = np.clip(u, pu.min() + 1e-5 * span, pu.max() - 1e-5 * span)
    q = query[:, None]
    u0, u1, v0, v1 = pu, np.roll(pu, -1), pv, np.roll(pv, -1)
    du = u1 - u0
    active = (abs(du) > 1e-7)[None] & (q >= np.minimum(u0, u1)[None]) & (q < np.maximum(u0, u1)[None])
    slopes = np.divide(v1 - v0, du, out=np.zeros_like(du), where=abs(du) > 1e-7)
    intersections = v0[None] + (q - u0[None]) * slopes[None]
    intersections = np.where(active, intersections, np.inf)
    if intersections.shape[1] % 2:
        intersections = np.pad(intersections, ((0, 0), (0, 1)), constant_values=np.inf)
        slopes = np.append(slopes, 0.)
    order = np.argsort(intersections, axis=1, kind='stable')
    sections = np.take_along_axis(intersections, order, 1)
    section_slopes = slopes[order]
    low, high = sections[:, 0::2], sections[:, 1::2]
    finite = np.isfinite(low) & np.isfinite(high)
    distance = np.where(finite, np.maximum(np.maximum(low - v[:, None], v[:, None] - high), 0), np.inf)
    selected = np.argmin(distance, axis=1)
    row = np.arange(len(pixels))
    lo, hi = low[row, selected], high[row, selected]
    valid = finite[row, selected] & (hi > lo)
    lo, hi = np.where(valid, lo, 0), np.where(valid, hi, 0)
    slope = .5 * (section_slopes[:, 0::2][row, selected] + section_slopes[:, 1::2][row, selected])
    midpoint, cross_height = .5 * (lo + hi), hi - lo
    height = cross_height / np.sqrt(1 + slope * slope)
    tangent = axis[None] + slope[:, None] * normal[None]
    angle = (np.arctan2(tangent[:, 1], tangent[:, 0]) + np.pi / 2) % np.pi - np.pi / 2
    return midpoint, height, angle, valid, cross_height


def targets_from_polygons(instances, image_hw, feature_hw, center_shrink=.3):
    """Single-image numpy maps: text/center int64; valid bool; height/angle float32.

    ``ignore=True`` masks ALL losses and wins in overlaps. Empty/### transcription
    does not imply ignore: text-only datasets still supervise detection.
    Degenerate zero-area polygons are skipped, not replaced by bounding boxes.
    Rasterization uses rounded feature-pixel vertices and an at-least-one-pixel
    center band so small legitimate words do not silently lose geometry targets.
    """
    ih, iw = map(int, image_hw)
    fh, fw = map(int, feature_hw)
    if min(ih, iw, fh, fw) <= 0 or not 0 < center_shrink <= 1:
        raise ValueError('Positive spatial sizes and center_shrink in (0,1] are required')
    target = dict(text=np.zeros((fh, fw), np.int64), center=np.zeros((fh, fw), np.int64),
                  valid=np.ones((fh, fw), bool), height=np.zeros((fh, fw), np.float32),
                  angle=np.zeros((fh, fw), np.float32))
    ignored = np.zeros((fh, fw), np.uint8)
    for instance in instances:
        polygon = np.asarray(instance['poly'], np.float32)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
            raise ValueError('Polygon must have at least three finite XY vertices')
        polygon = polygon * np.array([fw / iw, fh / ih], np.float32)
        if abs(cv2.contourArea(polygon)) < 1e-7:
            continue
        mask = np.zeros((fh, fw), np.uint8)
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
        if bool(instance.get('ignore', False)):
            ignored |= mask
            continue
        target['text'][mask > 0] = 1
        yy, xx = np.nonzero(mask)
        if not len(xx):
            continue
        pixels = np.stack([xx, yy], 1).astype(np.float32)
        axis = _principal_axis(pixels)
        midpoint, height, angle, valid, cross_height = _cross_sections(polygon, pixels, axis)
        normal = np.array([-axis[1], axis[0]])
        keep = valid & (abs(pixels @ normal - midpoint) <= np.maximum(.5 + 1e-6, center_shrink * cross_height / 2))
        cy, cx = yy[keep], xx[keep]
        target['center'][cy, cx] = 1
        target['height'][cy, cx] = height[keep]
        target['angle'][cy, cx] = angle[keep]
    target['valid'][ignored > 0] = False
    return target
