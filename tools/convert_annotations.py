"""Convert raw official annotations to the protocol jsonl (docs/02_dataset_protocol.md §1).
P0-02 / P0-01 (docs/03): raw/<name> -> data/annotations/<name>.jsonl, verifiable traceability.

Usage:
  python tools/convert_annotations.py --dataset lated     --raw data/raw --out data/annotations --seed 0
      -> latedspot_train.jsonl (900) / latedspot_val.jsonl (100, scene x illum stratified, fixed seed)
         latedspot_test.jsonl (official 500) / splits.json (frozen lists)
  python tools/convert_annotations.py --dataset ctw1500   --raw data/raw --out data/annotations [--split test|train]
  python tools/convert_annotations.py --dataset totaltext --raw data/raw --out data/annotations [--split test|train]
  python tools/convert_annotations.py --dataset synthtext --raw data/raw --out data/annotations
      -> synthtext_train.jsonl (wordBB 4-corner ring -> CTW convention, transcriptions kept)

LATeD/CTW detection-only labels use text="###" without opting out of detection.
Preserve explicit source ignore flags (including CTW1500 MMOCR don't-care);
missing transcription by itself does not imply ignore.
Instance/meta ids keep traceability to raw image + label line.
Polygons are normalised to the CTW1500 convention (top edge L->R, then bottom edge R->L):
ambiguous (near-vertical) polys are left unchanged and counted.
After conversion run: python tools/illum_stats.py --ann <file> --root data --out <tmp> (then replace).
"""
import argparse, json, os, re, sys, collections
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seeread.data.lowlight_synth import luminance, illum_bin

STATS = collections.Counter()


def normalize_ring(poly):
    """Ring (N,2) -> CTW convention [top L->R] + [bottom R->L]. Near-vertical polys: unchanged."""
    poly = np.asarray(poly, np.float32)
    if len(poly) < 4 or len(poly) % 2:
        STATS["poly_odd_left_as_is"] += 1
        return poly
    c1, c2 = poly[: len(poly) // 2], poly[len(poly) // 2:]
    my1, my2 = c1[:, 1].mean(), c2[:, 1].mean()
    h = max(poly[:, 1].max() - poly[:, 1].min(), 1e-3)
    if abs(my1 - my2) < 0.15 * h:                       # ambiguous orientation (near-vertical text)
        STATS["poly_ambiguous_left_as_is"] += 1
        return poly
    if my1 > my2:                                       # stored bottom-first: reverse both chains
        c1, c2 = c2[::-1].copy(), c1[::-1].copy()
        STATS["poly_chain_swapped"] += 1
    if c1[0, 0] > c1[-1, 0]:                            # top chain stored R->L: flip direction
        c1, c2 = c1[::-1].copy(), c2[::-1].copy()
        STATS["poly_dir_flipped"] += 1
    return np.concatenate([c1, c2], 0)


def poly_list(flat):
    if len(flat) < 6 or len(flat) % 2 or not all(np.isfinite(flat)):
        STATS["poly_invalid_dropped"] += 1
        return None
    pts = np.asarray(flat, np.float32).reshape(-1, 2)
    if len(pts) > 4 and np.linalg.norm(pts[0] - pts[-1]) < 1e-3:
        pts = pts[:-1]                                  # drop explicit closing point
    return normalize_ring(pts).round().astype(int).tolist()


def quad_ring(pts4):
    """SynthText wordBB 2x4 corner block -> CTW ring [tl,tr,br,bl] (top L->R, bottom R->L)."""
    pts = np.asarray(pts4, np.float32).T                # -> (4,2)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(d)], pts[np.argmax(d)]
    return np.stack([tl, tr, br, bl]).round().astype(int).tolist()


def make_row(image, instances, meta, lit=None):
    r = dict(image=image, instances=instances, meta=meta)
    if lit:
        r["lit"] = lit
    return r


def img_y(root, path):
    img = cv2.cvtColor(cv2.imread(os.path.join(root, path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return luminance(img)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def summary(name, rows):
    n_ins = sum(len(r["instances"]) for r in rows)
    n_ign = sum(bool(i.get("ignore")) for r in rows for i in r["instances"])
    n_det_only = sum(not i.get("ignore") and i["text"] in ("###", "") for r in rows for i in r["instances"])
    print(f"[{name}] images={len(rows)} instances={n_ins} ignore={n_ign} det_only={n_det_only}")


def qa_sheet(path, root, rows, k=12, tag=""):
    """Render sampled polys + centerlines (post-normalisation) for visual orientation check."""
    from seeread.data.geometry import split_polygon, resample_polyline
    rng = np.random.default_rng(0)
    picks = rng.choice(len(rows), size=min(k, len(rows)), replace=False)
    tiles = []
    for i in picks:
        r = rows[int(i)]
        img = cv2.imread(os.path.join(root, r["image"]))
        img = (np.clip(img.astype(np.float32) / 255 * 5, 0, 1) ** 0.5 * 255).astype(np.uint8)  # QA gamma boost
        for ins in r["instances"][:8]:
            p = np.asarray(ins["poly"], np.int32)
            cv2.polylines(img, [p], False, (0, 255, 0), 2)
            top, bot = split_polygon(np.asarray(ins["poly"], np.float32))
            cl = resample_polyline((top + bot[::-1]) / 2, 16).astype(int)
            cv2.polylines(img, [cl], False, (0, 0, 255), 2)
        h, w = img.shape[:2]
        sc = min(512 / w, 320 / h)
        img = cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))))
        canvas = np.zeros((320, 512, 3), np.uint8)
        canvas[: img.shape[0], : img.shape[1]] = img
        tiles.append(canvas)
    grid = np.concatenate([np.concatenate(tiles[: len(tiles) // 2], 1),
                           np.concatenate(tiles[len(tiles) // 2:], 1)], 0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, grid)
    print(f"qa-sheet: {path} ({tag})")


# ------------------------------------------------------------------ datasets
def conv_lated(raw, out, root, seed):
    src = os.path.join(raw, "LATeD")
    tr = [l.strip() for l in open(os.path.join(src, "train_list.txt")) if l.strip()]
    te = [l.strip() for l in open(os.path.join(src, "test_list.txt")) if l.strip()]
    rows = {}
    for stem in tr + te:
        rel = f"raw/LATeD/image/{stem}.jpg"
        assert os.path.exists(os.path.join(root, rel)), f"missing image {rel}"
        assert os.path.exists(os.path.join(src, "image_bright", f"{stem}.jpg")), f"missing bright {stem}"
        scene = stem.split("_")[0]
        ins = []
        for li, line in enumerate(open(os.path.join(src, "label", f"{stem}.txt"))):
            flat = [float(v) for v in line.replace(",", " ").split()]
            p = poly_list(flat)
            if p is None:
                STATS["lated_label_line_skipped"] += 1
                continue
            # confirmed text geometry without transcription: det GT (ignore=false), "###" only opts out of rec/e2e
            ins.append(dict(poly=p, text="###", lang="en", ignore=False, id=f"{stem}#{li}"))
        y = img_y(root, rel)
        rows[stem] = make_row(
            rel, ins,
            dict(source="LATeD", illum_kind="real", illum_y=round(y, 4), illum_bin=illum_bin(y),
                 scene=scene, id=stem),
            lit=f"raw/LATeD/image_bright/{stem}.jpg")
    # stratified 900/100: scene x illum_bin groups, seeded permutation, then trim/pad to exactly 100
    groups = collections.defaultdict(list)
    for s in sorted(tr):
        groups[(rows[s]["meta"]["scene"], rows[s]["meta"]["illum_bin"])].append(s)
    rng = np.random.default_rng(seed)
    val = []
    for g in sorted(groups):
        pool = groups[g]
        k = max(1, int(round(len(pool) * 0.1))) if len(pool) >= 5 else 0
        val += [pool[i] for i in rng.permutation(len(pool))[:k]]
    while len(val) > 100:                               # trim from the largest val group, deterministic
        g = max(sorted(groups), key=lambda g: (sum(s in val for s in groups[g]), g))
        val.remove(next(s for s in groups[g] if s in val))
    while len(val) < 100:                               # pad from largest train-only group, seeded
        g = max(sorted(groups), key=lambda g: (sum(s not in val for s in groups[g]), g))
        pool = groups[g]
        cand = [pool[i] for i in rng.permutation(len(pool)) if pool[i] not in val]
        val.append(cand[0])
    val = sorted(val)
    train = [s for s in sorted(tr) if s not in set(val)]
    write_jsonl(os.path.join(out, "latedspot_train.jsonl"), [rows[s] for s in train])
    write_jsonl(os.path.join(out, "latedspot_val.jsonl"), [rows[s] for s in val])
    write_jsonl(os.path.join(out, "latedspot_test.jsonl"), [rows[s] for s in sorted(te)])
    json.dump(dict(seed=seed, protocol="docs/02_dataset_protocol.md §5", source_train="train_list.txt(1000)",
                   train=train, val=val, test=sorted(te),
                   counts=dict(train=len(train), val=len(val), test=len(sorted(te)))),
              open(os.path.join(out, "splits.json"), "w"), indent=1)
    print(f"[lated] splits.json frozen: train={len(train)} val={len(val)} test={len(te)}")
    qa_sheet(os.path.join("results", "qa", "convert_lated.png"), root,
             [rows[s] for s in train[:6]] + [rows[s] for s in val[:6]],
             tag="lated train+val (green=ring, red=centerline)")


def conv_ctw1500(raw, out, root, split):
    base = os.path.join(raw, "CTW1500", "ctw1500")
    d = json.load(open(os.path.join(base, f"textdet_{split}.json")))
    rows = []
    for item in d["data_list"]:
        rel = "raw/CTW1500/ctw1500/" + item["img_path"]
        assert os.path.exists(os.path.join(root, rel)), f"missing image {rel}"
        ins = []
        for gi, g in enumerate(item["instances"]):
            p = poly_list(g["polygon"])
            if p is None:
                STATS["ctw_instance_skipped"] += 1
                continue
            # MMOCR's detection packer omits text but retains don't-care flags.
            ins.append(dict(poly=p, text="###", lang="en", ignore=bool(g.get("ignore", False)),
                            id=f"{os.path.basename(item['img_path'])}#{gi}"))
        rows.append(make_row(rel, ins, dict(source="CTW1500-mmocr", illum_kind="lit", scene="ctw",
                                            id=os.path.basename(item["img_path"]))))
    write_jsonl(os.path.join(out, f"ctw1500_{split}.jsonl"), rows)
    summary(f"ctw1500_{split}", rows)
    if split == "test":
        qa_sheet(os.path.join("results", "qa", "convert_ctw1500_test.png"), root, rows, tag="ctw1500 test")


TT_RE = re.compile(r"x: \[\[([\d\s.\-eE+]+)\]\], y: \[\[([\d\s.\-eE+]+)\]\], ornt: \[(?:u?['\"](.*?)['\"])?\], transcriptions: \[(.*)\]\s*$")


def _tt_texts(s):
    # A released label escapes the outer transcription quotes as u\'...\'.
    escaped = re.fullmatch(r"u\\'(.*)\\'", s.strip())
    if escaped:
        s = "u'" + escaped.group(1) + "'"
    return [a or b for a, b in re.findall(r"u?'([^']*)'|u?\"([^\"]*)\"", s)]


def conv_totaltext(raw, out, root, split):
    base = os.path.join(raw, "TotalText")
    gt_dir = os.path.join(base, "txt_format", split.capitalize())
    image_dir = os.path.join(base, "Images", split.capitalize())
    image_names = {os.path.splitext(name)[0]: name for name in os.listdir(image_dir)
                   if os.path.splitext(name)[1].lower() in (".jpg", ".jpeg", ".png")}
    names = sorted(f[8:-4] for f in os.listdir(gt_dir) if f.startswith("poly_gt_"))
    rows = []
    for nid in names:
        name = image_names.get(nid)
        if name is None:
            STATS["tt_missing_image_skipped"] += 1
            continue
        rel = f"raw/TotalText/Images/{split.capitalize()}/{name}"
        raw_lines = open(os.path.join(gt_dir, f"poly_gt_{nid}.txt")).read().splitlines()
        joined, buf = [], ""
        for ln in raw_lines:                            # re-join annotation-tool line wraps
            buf = buf + " " + ln.strip() if buf else ln.strip()
            if "transcriptions:" in buf:
                joined.append(buf); buf = ""
        if buf:
            joined.append(buf)
        ins = []
        for li, line in enumerate(joined):
            m = TT_RE.match(line.strip())
            if not m:
                STATS["tt_line_unparsed"] += 1
                continue
            xs, ys_, ornt, tr = m.groups()
            ornt = ornt or ""
            xy = list(zip([float(v) for v in xs.split()], [float(v) for v in ys_.split()]))
            p = poly_list([c for pt in xy for c in pt])
            if p is None:
                STATS["tt_instance_skipped"] += 1
                continue
            text = (_tt_texts(tr) + [""])[0]
            # Author column6 is the don't-care marker, independent of transcription.
            dontcare = ornt == "#"
            rec = dict(poly=p, text="###" if dontcare else text, lang="en", ignore=dontcare,
                       ornt=ornt, id=f"{nid}#{li}")
            if dontcare and text:
                rec["src"] = text
            ins.append(rec)
        rows.append(make_row(rel, ins, dict(source="TotalText", illum_kind="lit", scene="totaltext", id=nid)))
    write_jsonl(os.path.join(out, f"totaltext_{split}.jsonl"), rows)
    summary(f"totaltext_{split}", rows)
    qa_sheet(os.path.join("results", "qa", f"convert_totaltext_{split}.png"), root, rows, tag=f"totaltext {split}")


def conv_synthtext(raw, out, root):
    import scipy.io as sio
    base = os.path.join(raw, "SynthText", "SynthText")
    m = sio.loadmat(os.path.join(base, "gt.mat"), squeeze_me=True, struct_as_record=False)
    imnames, wordBB, txt = m["imnames"], m["wordBB"], m["txt"]
    n = len(imnames)
    rows = []
    for i in range(n):
        rel_img = str(imnames[i])
        rel = "raw/SynthText/SynthText/" + rel_img
        if not os.path.exists(os.path.join(root, rel)):
            STATS["st_missing_image_skipped"] += 1
            continue
        wb = np.asarray(wordBB[i])
        if wb.ndim != 3:                                # single word: (2,4) -> (2,4,1)
            wb = wb[..., None]
        t = txt[i]                                  # txt elements are LINES with space padding; flatten to words
        words = [w for x in ([t] if isinstance(t, np.str_) else np.atleast_1d(t)) for w in str(x).split()]
        nw = min(wb.shape[-1], len(words))
        if nw == 0 or wb.shape[-1] != len(words):
            STATS["st_text_box_mismatch"] += 1
        ins = []
        for w in range(nw):
            text = " ".join(words[w].split())           # collapse newlines/spaces
            if not text:
                continue
            p = quad_ring(wb[:2, :, w])
            ins.append(dict(poly=p, text=text, lang="en", ignore=False, id=f"{rel_img}#{w}"))
        if not ins:
            STATS["st_empty_image_skipped"] += 1
            continue
        scene = os.path.dirname(rel_img)
        rows.append(make_row(rel, ins, dict(source="SynthText", illum_kind="lit", scene=scene, id=rel_img)))
        if (i + 1) % 20000 == 0:
            print(f"  {i+1}/{n} images", flush=True)
    write_jsonl(os.path.join(out, "synthtext_train.jsonl"), rows)
    summary("synthtext_train", rows)
    qa_sheet(os.path.join("results", "qa", "convert_synthtext.png"), root, rows, tag="synthtext sample")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["lated", "ctw1500", "totaltext", "synthtext"])
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/annotations")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default="data")           # to resolve image paths for luminance
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.dataset == "lated":
        conv_lated(a.raw, a.out, a.root, a.seed)
    elif a.dataset == "ctw1500":
        conv_ctw1500(a.raw, a.out, a.root, a.split)
    elif a.dataset == "synthtext":
        conv_synthtext(a.raw, a.out, a.root)
    else:
        conv_totaltext(a.raw, a.out, a.root, a.split)
    print("poly normalisation:", dict(STATS))


if __name__ == "__main__":
    main()
