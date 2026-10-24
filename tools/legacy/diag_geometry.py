"""Why does a predicted-centerline crop fail even when polygon IoU is high?  (numpy only; no GPU, no training)
For every matched (pred, GT) pair, compare the predicted centerline+heights with the GT ones and assign ONE cause.
Dump from your attribution script, per matched pair:  pred_centers (K,2), pred_heights (K,), gt_centers (K,2), gt_heights (K,)
(all in the SAME coordinate frame, reading order as fed to rectify_strip), plus A_correct / B_correct booleans.
Then:  summarize([classify(...), ...], b_correct)  ->  share of B-failures per cause."""
from __future__ import annotations
from collections import Counter
import numpy as np


def _resample(c, k=16):
    c = np.asarray(c, np.float64)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    if s[-1] < 1e-9:
        return np.repeat(c[:1], k, 0), 0.0
    t = np.linspace(0, s[-1], k)
    return np.stack([np.interp(t, s, c[:, 0]), np.interp(t, s, c[:, 1])], 1), float(s[-1])


def classify(pc, ph, gc, gh, flip_ratio=0.8, h_lo=0.7, h_hi=1.4, drift=0.25, len_lo=0.85, len_hi=1.15):
    P, Lp = _resample(pc); G, Lg = _resample(gc)
    hg = float(np.median(gh)); hp = float(np.median(ph))
    d_fwd = float(np.linalg.norm(P - G, axis=1).mean()); d_rev = float(np.linalg.norm(P - G[::-1], axis=1).mean())
    Gd, _ = _resample(gc, 400)                          # dense GT polyline: lateral error = distance to the LINE, not index-wise
    lat = float(np.sqrt(((P[:, None] - Gd[None]) ** 2).sum(2)).min(1).mean())
    st = dict(flip_dist_ratio=d_rev / max(d_fwd, 1e-6), height_ratio=hp / max(hg, 1e-6), length_ratio=Lp / max(Lg, 1e-6),
              lateral_over_h=lat / max(hg, 1e-6))
    if d_rev < flip_ratio * d_fwd:                      cause = "flipped_order(180deg crop)"
    elif not (h_lo <= st["height_ratio"] <= h_hi):      cause = "height_wrong"
    elif st["lateral_over_h"] > drift:                  cause = "center_drift"
    elif st["length_ratio"] < len_lo:                   cause = "too_short(end truncation)"
    elif st["length_ratio"] > len_hi:                   cause = "too_long"
    else:                                               cause = "geometry_ok(recognizer brittle to crop shift)"
    return cause, st


def summarize(results, b_correct):
    """results: list of (cause, stats); b_correct: list[bool]. Prints share per cause among B-FAILURES and B accuracy per cause."""
    fails = Counter(c for (c, _), ok in zip(results, b_correct) if not ok)
    allc = Counter(c for c, _ in results)
    n_f = max(sum(fails.values()), 1)
    out = {c: dict(n=allc[c], b_fail_share=fails[c] / n_f, b_acc=1 - fails[c] / allc[c]) for c in allc}
    for c, v in sorted(out.items(), key=lambda kv: -kv[1]["b_fail_share"]):
        print(f"{c:48s} n={v['n']:4d}  share_of_B_failures={v['b_fail_share']:.2f}  B_acc_in_bucket={v['b_acc']:.2f}")
    return out


if __name__ == "__main__":                              # self-test on synthetic geometry
    K = 7; gc = np.stack([np.linspace(20, 220, K), np.full(K, 50.0)], 1); gh = np.full(K, 30.0)
    cases = {
        "flipped_order(180deg crop)": (gc[::-1], gh),
        "height_wrong": (gc, gh * 0.5),
        "center_drift": (gc + np.array([0, 14.0]), gh),
        "too_short(end truncation)": (gc[1:-1], gh[1:-1]),
        "too_long": (np.stack([np.linspace(0, 240, K), np.full(K, 50.0)], 1), gh),
        "geometry_ok(recognizer brittle to crop shift)": (gc + np.array([0, 1.0]), gh * 1.05),
    }
    for want, (pc, ph) in cases.items():
        got, _ = classify(pc, ph, gc, gh); assert got == want, (want, got)
    print("self-test OK: all 6 causes recognised"); summarize([classify(*v, gc, gh) for v in cases.values()], [False] * 6)
