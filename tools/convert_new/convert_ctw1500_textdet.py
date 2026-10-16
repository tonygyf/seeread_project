"""CTW1500 mmocr 'textdet' json (COCO style, polygons only) -> SeeRead jsonl.
This release has NO transcriptions: every instance becomes a DETECTION positive with text='###', ignore=false
(P0-14 semantics: contributes det targets only; never a recognition crop; counted in det recall at eval).
Because there are no transcripts, this set can only enter DETECTION training (S1.5 / S3 det branch), never the recognizer.
Polygon convention check: CTW1500 has 14 points = 7 top (L->R) + 7 bottom (R->L). Polygons whose first->middle point runs
right-to-left (i.e. rotated text) are kept; we only REPORT how many look reversed so you can inspect a few by eye.
python tools/convert_ctw1500_textdet.py --json data/raw/CTW1500/ctw1500/textdet_train.json --img_dir data/raw/CTW1500/ctw1500 --root data --out data/annotations/ctw1500_train_det.jsonl"""
import argparse, json, os
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True); ap.add_argument("--img_dir", required=True)
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(a.json, encoding="utf-8"))
    # mmocr textdet format: {metainfo, data_list:[{img_path, height, width,
    #   instances:[{polygon: flat 28, bbox, bbox_label, ignore}]}]}
    n_img = n_ins = n_rev = n_flip = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for it in d["data_list"]:
            path = os.path.join(a.img_dir, it["img_path"])   # img_path is relative to the mmocr data root (e.g. textdet_imgs/train/0744.jpg)
            inst = []
            for an in it["instances"]:
                seg = an["polygon"]
                if len(seg) < 8:
                    continue
                poly = [[float(seg[i]), float(seg[i + 1])] for i in range(0, len(seg) - 1, 2)]
                h = len(poly) // 2
                # Normalize order to top-L->R / bottom-R->L: flip ONLY near-horizontal reversed tops
                # (curved text legitimately runs R->L in image space; flipping it would corrupt the
                #  direction supervision, so leave it annotated along its reading direction).
                dx, dy = poly[h - 1][0] - poly[0][0], poly[h - 1][1] - poly[0][1]
                if dx < -1e-6 and abs(dx) > 2 * abs(dy):
                    poly = poly[:h][::-1] + poly[h:][::-1]   # reverse each half: top becomes L->R, bottom R->L
                    n_flip += 1
                dx2, dy2 = poly[h - 1][0] - poly[0][0], poly[h - 1][1] - poly[0][1]
                n_rev += int(dx2 < -1e-6 and abs(dx2) > abs(dy2))
                inst.append(dict(poly=poly, text="###", ignore=bool(an.get("ignore", 0)), lang="en"))
                n_ins += 1
            if not inst:
                continue
            n_img += 1
            f.write(json.dumps(dict(image=os.path.relpath(path, a.root), lit=None, instances=inst,
                                    meta=dict(illum_kind="lit", source="ctw1500_train")), ensure_ascii=False) + "\n")
    print(f"images={n_img} instances={n_ins} flipped-to-LtR={n_flip} remaining-RtL-tops(curved,ok)={n_rev}")


if __name__ == "__main__":
    main()
