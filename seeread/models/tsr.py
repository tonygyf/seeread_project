"""TSR: bottom-up text shaping by Rotated-rectangle accumulation with Farthest Point Sampling (inference, numpy/cv2).
NOTE: Eq. (6) of the paper is ambiguous ("farthest from any point in T"); we implement standard FPS
(farthest from the already-selected set) -- verify against the official code."""
from __future__ import annotations
import numpy as np, cv2


def farthest_point_sampling(pts: np.ndarray, k: int) -> np.ndarray:
    n = len(pts); k = min(k, n)
    first = int(np.argmax(((pts - pts.mean(0)) ** 2).sum(1)))
    sel, d = [first], ((pts - pts[first]) ** 2).sum(1)
    for _ in range(k - 1):
        i = int(np.argmax(d)); sel.append(i)
        d = np.minimum(d, ((pts - pts[i]) ** 2).sum(1))
    return np.array(sel)


def order_along_principal_axis(pts):
    """Order points along the principal axis, reading direction = left->right (or top->bottom for vertical text)."""
    c = pts - pts.mean(0)
    ax = np.linalg.svd(c, full_matrices=False)[2][0]
    if (abs(ax[0]) >= abs(ax[1]) and ax[0] < 0) or (abs(ax[0]) < abs(ax[1]) and ax[1] < 0):
        ax = -ax
    return np.argsort(c @ ax)


def tsr_decode(text_p, center_p, geo, stride=4, thr_center=0.5, thr_text=0.5, fixed_w=6.0,
               step=3.0, min_area=8, k_max=32, close=5):
    """text_p, center_p: HxW probs; geo: 3xHxW (log1p h, cos, sin) at feature res. Returns list of dicts in input px."""
    Hf, Wf = text_p.shape
    cmask = ((center_p > thr_center) & (text_p > thr_text)).astype(np.uint8)
    n, lab = cv2.connectedComponents(cmask, connectivity=8)
    res = []
    for cid in range(1, n):
        ys, xs = np.nonzero(lab == cid)
        if len(ys) < min_area:
            continue
        pts = np.stack([xs, ys], 1).astype(np.float32)
        ext = np.linalg.norm(pts.max(0) - pts.min(0)) + 1
        sel = farthest_point_sampling(pts, int(np.clip(np.ceil(ext / step) + 1, 2, k_max)))
        P = pts[sel]
        acc = np.zeros((Hf, Wf), np.uint8)
        hs = []
        for (x, y) in P:
            xi, yi = int(x), int(y)
            h = float(np.expm1(max(geo[0, yi, xi], 0.0)))
            d = np.array([geo[1, yi, xi], geo[2, yi, xi]], np.float32)
            d = d / (np.linalg.norm(d) + 1e-6)
            nrm = np.array([-d[1], d[0]], np.float32)
            c = np.array([x, y], np.float32)
            quad = np.stack([c - d * fixed_w / 2 - nrm * h / 2, c + d * fixed_w / 2 - nrm * h / 2,
                             c + d * fixed_w / 2 + nrm * h / 2, c - d * fixed_w / 2 + nrm * h / 2])
            cv2.fillConvexPoly(acc, quad.round().astype(np.int32), 1)
            hs.append(h)
        acc = cv2.morphologyEx(acc, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
        cnts, _ = cv2.findContours(acc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < min_area:
            continue
        poly = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True).reshape(-1, 2).astype(np.float32)
        order = order_along_principal_axis(P)
        m = np.zeros((Hf, Wf), np.uint8); cv2.fillPoly(m, [cnt], 1)
        res.append(dict(poly=poly * stride, score=float(text_p[m > 0].mean()),
                        centers=P[order] * stride, heights=np.array(hs, np.float32)[order] * stride))
    return res
