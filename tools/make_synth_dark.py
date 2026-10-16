"""Generate FIXED-SEED synthetic dark test sets (Dark-Total-Text / Dark-CTW1500) at 3 levels, keeping lit pairs.
python tools/make_synth_dark.py --ann data/annotations/ctw1500_test.jsonl --root data --out data/dark_ctw1500_test --seed 1234"""
import argparse, json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seeread.data.lowlight_synth import darken, PHYS_LEVELS, illum_bin, luminance

ap = argparse.ArgumentParser(); ap.add_argument("--ann", required=True); ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--mode", default="physical")
a = ap.parse_args(); os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
rows = [json.loads(l) for l in open(a.ann, encoding="utf-8") if l.strip()]
with open(os.path.join(a.out, "ann.jsonl"), "w", encoding="utf-8") as f:
    for i, r in enumerate(rows):
        img = cv2.cvtColor(cv2.imread(os.path.join(a.root, r["image"])), cv2.COLOR_BGR2RGB)
        for lv in PHYS_LEVELS:
            rng = np.random.default_rng((a.seed, i, list(PHYS_LEVELS).index(lv))) 
            dk, meta = darken(img, rng, level=lv, mode=a.mode)
            name = f"images/{i:05d}_{lv}.png"; cv2.imwrite(os.path.join(a.out, name), cv2.cvtColor(dk, cv2.COLOR_RGB2BGR))
            meta["illum_kind"] = "synth"
            dk01 = dk.astype(np.float32) / 255.0
            insts = []
            for ins in r["instances"]:                   # re-derive per-instance bin from the DARK pixels (same bbox rule as illum_stats)
                ins = dict(ins)
                if "bin" in ins:
                    ins["bin_source"] = ins.pop("bin")   # lit-source bin kept under a separate field
                p = np.asarray(ins["poly"]); x0, y0 = np.maximum(p.min(0).astype(int), 0); x1, y1 = p.max(0).astype(int) + 1
                patch = dk01[y0:y1, x0:x1]
                ins["bin"] = illum_bin(luminance(patch)) if patch.size else meta["illum_bin"]
                insts.append(ins)
            f.write(json.dumps(dict(image=os.path.relpath(os.path.join(a.out, name), a.root), lit=r["image"], instances=insts, meta=meta), ensure_ascii=False) + "\n")
