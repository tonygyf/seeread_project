"""ICDAR2015 Incidental (Ch4) train/test -> SeeRead jsonl.
Input : <img_dir>/img_N.jpg  and  <gt_dir>/gt_img_N.txt  with lines  x1,y1,x2,y2,x3,y3,x4,y4,transcription  (utf-8 BOM possible)
Output: one json per image. '###' = official don't-care -> ignore=true (excluded from det targets AND rec, absorbs predictions in eval).
Quad order is kept as annotated (clockwise from the text start), which matches the top-L->R / bottom-R->L convention.
Images are well-lit: illum_kind='lit' so they are darkened ONLINE in training (and keep their lit pair for CIC).
python tools/convert_icdar2015.py --img_dir data/raw/ICDAR2015/ch4_training_images --gt_dir data/raw/ICDAR2015/ch4_training_localization_transcription_gt --root data --out data/annotations/icdar2015_train.jsonl"""
import argparse, glob, json, os, re


def parse_gt(path):
    inst = []
    for line in open(path, encoding="utf-8-sig"):
        line = line.strip()
        if not line:
            continue
        p = line.split(",", 8)                      # transcription may itself contain commas
        if len(p) < 9:
            continue
        xy = [int(float(v)) for v in p[:8]]
        text = p[8].strip()
        dc = text == "###"
        inst.append(dict(poly=[[xy[i], xy[i + 1]] for i in range(0, 8, 2)], text=text, ignore=dc, lang="en"))
    return inst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True); ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n_img = n_ins = n_dc = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for ip in sorted(glob.glob(os.path.join(a.img_dir, "*.jpg")) + glob.glob(os.path.join(a.img_dir, "*.png"))):
            m = re.search(r"img_(\d+)", os.path.basename(ip))
            gp = os.path.join(a.gt_dir, f"gt_img_{m.group(1)}.txt") if m else None
            if not gp or not os.path.exists(gp):
                continue
            inst = parse_gt(gp)
            n_img += 1; n_ins += len(inst); n_dc += sum(i["ignore"] for i in inst)
            f.write(json.dumps(dict(image=os.path.relpath(ip, a.root), lit=None, instances=inst,
                                    meta=dict(illum_kind="lit", source="icdar2015")), ensure_ascii=False) + "\n")
    print(f"images={n_img} instances={n_ins} dont_care={n_dc} transcribed={n_ins - n_dc}")


if __name__ == "__main__":
    main()
