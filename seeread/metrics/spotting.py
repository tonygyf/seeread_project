"""Detection / recognition / end-to-end spotting metrics with illumination-stratified reporting.
Protocol: greedy matching by IoU>=0.5 (CTW1500-style); don't-care GT ('###' / ignore) absorbs predictions;
E2E = matched AND normalised-text equal ("None" lexicon). Per-bin numbers are RECALL-style (precision is not binnable)."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from shapely.geometry import Polygon


def norm_text(s: str) -> str:
    return "".join(c for c in s if c.isalnum()).lower()


def edit_distance(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(b) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[-1]


def _poly(p):
    q = Polygon(np.asarray(p, np.float64).reshape(-1, 2))
    return q if q.is_valid else q.buffer(0)


def iou(a, b):
    try:
        i = a.intersection(b).area
        return i / (a.union(b).area + 1e-9)
    except Exception:
        return 0.0


class SpottingEvaluator:
    def __init__(self, iou_thr=0.5):
        self.thr = iou_thr
        self.c = defaultdict(lambda: defaultdict(float))   # bin -> counters
        self.c_all = self.c["all"]

    def add(self, preds, gts, image_bin=None):
        """preds: [{'poly','text','score'}]; gts: [{'poly','text','ignore'?,'bin'?}] in the SAME coordinate frame.

        ignore=true  : geometry don't-care (TotalText '#'); matched preds absorbed, no GT counting.
        text '###'/'' with ignore=false: confirmed text without transcription (LATeD/CTW): counts for
        detection only, never for e2e (mirrors the P0-13 training-side split)."""
        gp = [_poly(g["poly"]) for g in gts]
        dont = [bool(g.get("ignore")) for g in gts]
        texted = [g.get("text", "") not in ("###", "") for g in gts]
        order = sorted(range(len(preds)), key=lambda i: -preds[i].get("score", 1.0))
        used, n_pred = set(), 0
        for i in order:
            pp = _poly(preds[i]["poly"])
            best, bj = 0.0, -1
            for j, g in enumerate(gp):
                if j in used:
                    continue
                v = iou(pp, g)
                if v > best:
                    best, bj = v, j
            if best >= self.thr and bj >= 0:
                used.add(bj)
                if dont[bj]:
                    continue                                # absorbed by geometry don't-care
                n_pred += 1
                g = gts[bj]; key = g.get("bin") or image_bin or "unk"
                for k in ("all", key):
                    self.c[k]["tp_det"] += 1                 # valid text geometry (transcribed or not)
                if texted[bj]:
                    ok = norm_text(preds[i]["text"]) == norm_text(g["text"])
                    ref = norm_text(g["text"])
                    for k in ("all", key):
                        self.c[k]["tp_e2e"] += ok; self.c[k]["n_match_e2e"] += 1
                        self.c[k]["edits"] += edit_distance(norm_text(preds[i]["text"]), ref)
                        self.c[k]["chars"] += max(len(ref), 1)
            else:
                n_pred += 1
        self.c_all["n_pred"] += n_pred
        for j, g in enumerate(gts):
            if dont[j]:
                continue
            key = g.get("bin") or image_bin or "unk"
            for k in ("all", key):
                self.c[k]["n_gt"] += 1                      # detection denominator includes textless GT
                if j not in used:
                    self.c[k]["missed"] += 1
                if texted[j]:
                    self.c[k]["n_gt_e2e"] += 1              # e2e denominator: transcribed GT only

    def summary(self):
        res = {}
        for k, c in self.c.items():
            n_gt, tp, e2e = c["n_gt"], c["tp_det"], c["tp_e2e"]
            n_gt_e2e, n_match_e2e = c["n_gt_e2e"], c["n_match_e2e"]
            r = dict(n_gt=int(n_gt), n_gt_e2e=int(n_gt_e2e),
                     det_recall=tp / max(n_gt, 1), e2e_recall=e2e / max(n_gt_e2e, 1),
                     rec_acc_on_det=e2e / max(n_match_e2e, 1),
                     cer_on_det=c["edits"] / max(c["chars"], 1),
                     frac_missed=c["missed"] / max(n_gt, 1), frac_misread=(n_match_e2e - e2e) / max(n_gt_e2e, 1))
            if k == "all":
                npred = c["n_pred"]; p_det, p_e2e = tp / max(npred, 1), e2e / max(npred, 1)
                r.update(det_P=p_det, det_R=r["det_recall"], det_F=2 * p_det * r["det_recall"] / max(p_det + r["det_recall"], 1e-9),
                         e2e_P=p_e2e, e2e_R=r["e2e_recall"], e2e_F=2 * p_e2e * r["e2e_recall"] / max(p_e2e + r["e2e_recall"], 1e-9))
            res[k] = r
        return res
