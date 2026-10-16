"""Polygon utilities + detector GT maps (text / center / height / direction)."""
from __future__ import annotations
import numpy as np, cv2


def canonical_quad(pts):
    box = cv2.boxPoints(cv2.minAreaRect(np.asarray(pts, np.float32)))
    s, d = box.sum(1), np.diff(box, axis=1).ravel()
    return np.stack([box[np.argmin(s)], box[np.argmin(d)], box[np.argmax(s)], box[np.argmax(d)]]).astype(np.float32)  # tl,tr,br,bl


def split_polygon(poly):
    """CTW1500 convention: top side left->right, then bottom side right->left. Returns top, bottom both L->R.

    Canonicalizes reading direction: some annotations trace the top side right->left (or bottom->top for
    vertical text).  Without a fix this yields a 180deg-reversed centerline, injecting direction noise into the
    det direction head AND producing upside-down recognition crops (measured at ~7% of TotalText GT)."""
    poly = np.asarray(poly, np.float32)
    if len(poly) < 4 or len(poly) % 2:
        poly = canonical_quad(poly)
    n = len(poly) // 2
    top, bot = poly[:n], poly[n:][::-1].copy()
    if len(top) >= 2:
        c = top - top.mean(0)
        ax = np.linalg.svd(c, full_matrices=False)[2][0]
        if (abs(ax[0]) >= abs(ax[1]) and ax[0] < 0) or (abs(ax[0]) < abs(ax[1]) and ax[1] < 0):
            ax = -ax                                   # canonical reading axis: +x (horizontal) / +y (vertical)
        if float((top[-1] - top[0]) @ ax) < 0:          # polygon traced against reading order -> flip both halves
            top, bot = top[::-1].copy(), bot[::-1].copy()
    return top, bot


def resample_polyline(pts, k):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return np.repeat(pts[:1], k, 0).astype(np.float32)
    t = np.linspace(0, s[-1], k)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], 1).astype(np.float32)


def centerline_and_height(poly, k=7):
    top, bot = split_polygon(poly)
    top, bot = resample_polyline(top, k), resample_polyline(bot, k)
    return (top + bot) / 2, np.linalg.norm(top - bot, axis=1)


def make_det_targets(h, w, instances, stride=4, shrink=0.35, k=8):
    """instances: dicts with 'poly' (network-input px), optional 'ignore'. Height is in feature-map pixels."""
    Hf, Wf = h // stride, w // stride
    z = lambda: np.zeros((Hf, Wf), np.float32)
    text, center, hmap, cosm, sinm, ign = z(), z(), z(), z(), z(), z()
    for ins in instances:
        poly = np.asarray(ins["poly"], np.float32) / stride
        pi = poly.round().astype(np.int32)
        if ins.get("ignore"):
            cv2.fillPoly(ign, [pi], 1.0)
            continue
        cv2.fillPoly(text, [pi], 1.0)
        top, bot = split_polygon(poly)
        top, bot = resample_polyline(top, k), resample_polyline(bot, k)
        c = (top + bot) / 2
        for i in range(k - 1):
            t = c[i + 1] - c[i]
            L = np.linalg.norm(t)
            if L < 1e-6:
                continue
            t = t / L
            hh = 0.5 * (np.linalg.norm(top[i] - bot[i]) + np.linalg.norm(top[i + 1] - bot[i + 1]))
            q = np.stack([c[i] + (top[i] - c[i]) * shrink, c[i + 1] + (top[i + 1] - c[i + 1]) * shrink,
                          c[i + 1] + (bot[i + 1] - c[i + 1]) * shrink, c[i] + (bot[i] - c[i]) * shrink])
            qi = q.round().astype(np.int32)
            cv2.fillPoly(center, [qi], 1.0)
            cv2.fillPoly(hmap, [qi], float(hh))
            cv2.fillPoly(cosm, [qi], float(t[0]))
            cv2.fillPoly(sinm, [qi], float(t[1]))
    return dict(text=text, center=center, height=hmap, cos=cosm, sin=sinm, weight=1.0 - ign)
