"""Fill meta.illum_y / illum_bin per image AND per instance ('bin' key) for a real dataset; print the histogram used to
calibrate BIN_EDGES and the synthetic levels.   python tools/illum_stats.py --ann a.jsonl --root data --out a_stat.jsonl"""
import argparse, json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seeread.data.lowlight_synth import luminance, illum_bin

ap = argparse.ArgumentParser(); ap.add_argument("--ann", required=True); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); ys = []
with open(a.out, "w", encoding="utf-8") as f:
    for l in open(a.ann, encoding="utf-8"):
        r = json.loads(l); img = cv2.cvtColor(cv2.imread(os.path.join(a.root, r["image"])), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        y = luminance(img); r.setdefault("meta", {}).update(illum_y=y, illum_bin=illum_bin(y)); ys.append(y)
        for ins in r["instances"]:           # instance-level luminance = mean inside the polygon's bounding box
            p = np.asarray(ins["poly"]); x0, y0 = np.maximum(p.min(0).astype(int), 0); x1, y1 = p.max(0).astype(int) + 1
            patch = img[y0:y1, x0:x1]; ins["bin"] = illum_bin(luminance(patch)) if patch.size else r["meta"]["illum_bin"]
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("image-level Y percentiles [5,25,50,75,95]:", np.percentile(ys, [5, 25, 50, 75, 95]).round(4))
