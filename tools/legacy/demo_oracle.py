"""Oracle end-to-end demo WITHOUT any training (qualitative pipeline walkthrough):
GT boxes (oracle detector) -> centerline rectification -> PRETRAINED TrOCR (frozen) -> compare vs GT text.
Two datasets:
  * ICDAR2015 incidental scene text (test 500 imgs): full scene -> GT quad -> rectified strip, plus an
    ONLINE physical low-light synthesis pass (seeread.data.lowlight_synth.darken, level=dark) to show the
    same pipeline on darkened input.
  * ESTR word crops: well_lit vs low_light pair recognition vs label.
Outputs per-sample bands + composites + summary.json under --out.
Usage: python tools/legacy/demo_oracle.py --icdar-root data/raw/ICDAR2015 --estr-root data/raw/ESTR \
           --out results/demo --n 20 --estr-n 20 --seed 0
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, cv2, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from seeread.data.geometry import centerline_and_height
from seeread.data.lowlight_synth import darken
from seeread.metrics.spotting import norm_text, edit_distance
from seeread.models.rectify import rectify_strip
from seeread.models.recognizer import TrOCRRecognizer

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, RED, YELLOW, BLUE, BLACK, GRAY = (40, 180, 40), (0, 0, 230), (0, 170, 230), (220, 120, 0), (30, 30, 30), (120, 120, 120)


def sanitize(s, maxlen=34):
    s = "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in str(s))
    return (s[: maxlen - 1] + "~") if len(s) > maxlen else s


def put(img, txt, xy, color=BLACK, scale=0.45, th=1):
    cv2.putText(img, txt, xy, FONT, scale, (255, 255, 255), th + 2, cv2.LINE_AA)
    cv2.putText(img, txt, xy, FONT, scale, color, th, cv2.LINE_AA)


def load_rgb(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def strips_for(img_u8, polys, crop_hw):
    """GT-quad oracle: rectify one strip per polygon. Returns (N,3,Ho,Wo) in [0,1] and kept instances."""
    t = torch.from_numpy(np.ascontiguousarray(img_u8)).permute(2, 0, 1)[None].float() / 255.0
    out = []
    for poly in polys:
        c, h = centerline_and_height(np.asarray(poly, np.float32), k=7)
        s = rectify_strip(t, torch.from_numpy(c).float(), torch.from_numpy(h).float(), out_hw=tuple(crop_hw))
        out.append(s[0])
    return torch.stack(out) if out else torch.zeros(0, 3, *crop_hw)


def predict(rec, crops, bs=16):
    if len(crops) == 0:
        return []
    preds = []
    for i in range(0, len(crops), bs):
        preds += rec.predict(crops[i : i + bs].cuda())
    return preds


# ---------------------------------------------------------------- ICDAR2015
def load_icdar_gt(root, img_id):
    out = []
    with open(f"{root}/gt_img_{img_id}.txt", encoding="utf-8-sig", errors="replace") as f:
        for ln in f:
            parts = ln.strip().split(",", 8)
            if len(parts) < 9:
                continue
            text = parts[8].strip()
            if text == "###" or not norm_text(text):
                continue
            out.append(dict(poly=np.array(list(map(float, parts[:8])), np.float32).reshape(4, 2), text=text))
    return out


def valid_instances(insts):
    keep = []
    for ins in insts:
        c, h = centerline_and_height(ins["poly"], k=7)
        length = float(np.linalg.norm(c[-1] - c[0]))
        if h.mean() >= 10 and length >= 16 and 1 <= len(norm_text(ins["text"])) <= 24:
            ins["height"] = float(h.mean())
            keep.append(ins)
    return keep


def draw_scene(img_rgb, insts, colors):
    vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()
    H, W = vis.shape[:2]
    scale = 240 / H
    vis = cv2.resize(vis, (max(int(W * scale), 40), 240), interpolation=cv2.INTER_AREA)
    for ins, col in zip(insts, colors):
        pts = (ins["poly"] * scale).round().astype(np.int32)
        cv2.polylines(vis, [pts], True, col, 2, cv2.LINE_AA)
    return vis


def make_band(img_rgb, dark_rgb, insts, pred_lit, pred_dark, meta, title):
    """Columns: [original+GT boxes | darkened+match boxes | lit strips+pred | dark strips+pred]."""
    def verdict(p, g):
        np_, ng = norm_text(p), norm_text(g)
        if np_ == ng:
            return GREEN, "OK"
        return (YELLOW, "PART") if edit_distance(np_, ng) <= max(1, len(ng) // 2) else (RED, "MISS")

    v_lit = [verdict(p, i["text"]) for p, i in zip(pred_lit, insts)]
    v_dark = [verdict(p, i["text"]) for p, i in zip(pred_dark, insts)]
    col_scene_o = draw_scene(img_rgb, insts, [GREEN] * len(insts))
    col_scene_d = draw_scene(dark_rgb, insts, [c for c, _ in v_dark])
    lit_col = strip_column([s.numpy() for s in lit_strips_cache], insts, pred_lit, v_lit)
    dark_col = strip_column([s.numpy() for s in dark_strips_cache], insts, pred_dark, v_dark)

    cols = [("original + GT boxes", col_scene_o), ("online darkened", col_scene_d),
            ("rectified strips (lit) + TrOCR pred", lit_col), ("rectified strips (dark) + pred", dark_col)]
    header, GAP = 24, 8
    H = max(c.shape[0] for _, c in cols) + header
    W = sum(c.shape[1] for _, c in cols) + GAP * (len(cols) + 1)
    band = np.full((H + 26, W, 3), 255, np.uint8)
    put(band, sanitize(title, 120), (GAP, 16), BLUE, 0.5, 1)
    x = GAP
    for name, c in cols:
        put(band, name, (x, header - 6), GRAY, 0.42, 1)
        band[header : header + c.shape[0], x : x + c.shape[1]] = c
        x += c.shape[1] + GAP
    stats = f"instances={len(insts)}  lit exact {sum(c == 'OK' for _, c in v_lit)}/{len(insts)}  dark exact {sum(c == 'OK' for _, c in v_dark)}/{len(insts)}"
    put(band, stats + f"   ({sanitize(meta, 70)})", (GAP, H + 14), BLACK, 0.45, 1)
    return band


def strip_column(strips, insts, preds, verdicts, sw=256, sh=64, row_h=None):
    """One column: per-instance [strip image | GT / pred text]."""
    row_h = row_h or (sh + 8)
    pad, tx = 4, sw + 8
    H, W = row_h * len(strips) + pad, sw + 240 + pad
    col = np.full((max(H, 60), W, 3), 255, np.uint8)
    y = pad
    for s, ins, p, (vc, tag) in zip(strips, insts, preds, verdicts):
        u8 = (np.clip(s.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        u8 = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
        col[y : y + sh, pad : pad + sw] = cv2.resize(u8, (sw, sh), interpolation=cv2.INTER_AREA)
        put(col, f"GT {sanitize(ins['text'])}", (tx, y + 16), BLACK, 0.42, 1)
        put(col, f"P  {sanitize(p)}  [{tag}]", (tx, y + 40), vc, 0.42, 1)
        y += row_h
    return col


def crops_estr(img_u8):
    return torch.from_numpy(np.ascontiguousarray(img_u8)).permute(2, 0, 1).float() / 255.0


# ---------------------------------------------------------------- composites
def compose_estr(rows, path, top=6):
    lit_ok_dark_bad = [i for i, r in enumerate(rows) if r["ok_lit"] and not r["ok_dark"]]
    both_ok = [i for i, r in enumerate(rows) if r["ok_lit"] and r["ok_dark"]]
    rest = [i for i in range(len(rows)) if i not in lit_ok_dark_bad and i not in both_ok]
    sel = [rows[i] for i in (lit_ok_dark_bad + both_ok + rest)[:top]]
    H0 = 96
    row_imgs = []
    for r in sel:
        cells = []
        for img, pred, col, tag in [(r["lit"], r["p_lit"], GREEN if r["ok_lit"] else RED, "well_lit"),
                                    (r["dark"], r["p_dark"], GREEN if r["ok_dark"] else RED, "low_light")]:
            h, w = img.shape[:2]
            s = min(H0 / h, 3.0)
            cell = cv2.copyMakeBorder(cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                                 (max(int(w * s), 24), int(h * s)), interpolation=cv2.INTER_AREA),
                                      2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(200, 200, 200))
            cells.append((cell, f"{tag}: {sanitize(pred)}", col))
        cw = max(c[0].shape[1] for c in cells) + 12
        TW = 300
        row = np.full((H0 + 34, cw * 2 + TW, 3), 255, np.uint8)
        x = 4
        for cell, txt, col in cells:
            row[4 : 4 + cell.shape[0], x : x + cell.shape[1]] = cell
            put(row, txt, (x, H0 + 26), col, 0.45, 1)
            x += cw
        put(row, f"GT: {sanitize(r['label'])}", (x, 30), BLACK, 0.5, 1)
        put(row, f"lit  pred: {sanitize(r['p_lit'])}", (x, 58), GREEN if r["ok_lit"] else RED, 0.5, 1)
        put(row, f"dark pred: {sanitize(r['p_dark'])}", (x, 86), GREEN if r["ok_dark"] else RED, 0.5, 1)
        row_imgs.append(row)
    W = max(r.shape[1] for r in row_imgs) + 16
    H = sum(r.shape[0] for r in row_imgs) + 40 + 8 * len(row_imgs)
    canvas = np.full((H, W, 3), 255, np.uint8)
    put(canvas, "ESTR test pairs | frozen pretrained TrOCR | same word crop: well_lit vs low_light", (8, 24), BLUE, 0.55, 1)
    y = 40
    for r in row_imgs:
        canvas[y : y + r.shape[0], 8 : 8 + r.shape[1]] = r
        y += r.shape[0] + 8
    cv2.imwrite(path, canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icdar-root", default="data/raw/ICDAR2015")
    ap.add_argument("--estr-root", default="data/raw/ESTR")
    ap.add_argument("--out", default="results/demo")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--estr-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}; loading frozen pretrained TrOCR (microsoft/trocr-base-printed)...")
    rec = TrOCRRecognizer(freeze=True).to(dev).eval()
    crop_hw = (64, 256)

    global lit_strips_cache, dark_strips_cache
    icdar_bands, icdar_meta, icdar_rec = [], [], []
    ids = sorted(int(f[4:-4]) for f in os.listdir(a.icdar_root) if f.startswith("img_") and f.endswith(".jpg"))
    cand = [(i, valid_instances(load_icdar_gt(a.icdar_root, i))) for i in ids]
    cand = [(i, ins) for i, ins in cand if 1 <= len(ins) <= 6]
    rng = np.random.default_rng(a.seed)
    pick = sorted((cand[i] for i in rng.choice(len(cand), size=min(a.n, len(cand)), replace=False)))
    for k, (img_id, insts) in enumerate(pick):
        img = load_rgb(f"{a.icdar_root}/img_{img_id}.jpg")
        polys = [ins["poly"] for ins in insts]
        lit_strips_cache = strips_for(img, polys, crop_hw)
        dark, meta = darken(img, np.random.default_rng([a.seed, img_id]), level="dark", mode="physical")
        dark_strips_cache = strips_for(dark, polys, crop_hw)
        pred_lit, pred_dark = predict(rec, lit_strips_cache), predict(rec, dark_strips_cache)
        meta_s = f"darken physical/{meta['level']} y={meta['illum_y']:.3f} bin={meta['illum_bin']}"
        band = make_band(img, dark, insts, pred_lit, pred_dark, meta_s, f"ICDAR2015 img_{img_id}.jpg   [#{k + 1}/{len(pick)}]")
        icdar_bands.append(band)
        ok_lit = [norm_text(p) == norm_text(ins["text"]) for p, ins in zip(pred_lit, insts)]
        ok_dark = [norm_text(p) == norm_text(ins["text"]) for p, ins in zip(pred_dark, insts)]
        icdar_meta.append(dict(img=f"img_{img_id}", n=len(insts), ok_lit=any(ok_lit), ok_dark=any(ok_dark)))
        icdar_rec += [dict(image=f"img_{img_id}", gt=ins["text"], pred_lit=pred_lit[j], pred_dark=pred_dark[j],
                           exact_lit=ok_lit[j], exact_dark=ok_dark[j],
                           cer_lit=edit_distance(norm_text(pred_lit[j]), norm_text(ins["text"])) / max(len(norm_text(ins["text"])), 1),
                           cer_dark=edit_distance(norm_text(pred_dark[j]), norm_text(ins["text"])) / max(len(norm_text(ins["text"])), 1))
                      for j, ins in enumerate(insts)]
        print(f"[icdar {k + 1}/{len(pick)}] img_{img_id}: {len(insts)} inst  lit {sum(ok_lit)}/{len(insts)}  dark {sum(ok_dark)}/{len(insts)}")
    os.makedirs(f"{a.out}/bands", exist_ok=True)
    for b, (img_id, _) in zip(icdar_bands, pick):
        cv2.imwrite(f"{a.out}/bands/icdar_img_{img_id}.png", b)

    estr_rows, estr_rec = [], []
    items = [json.loads(l) for l in open(f"{a.estr_root}/test.jsonl", encoding="utf-8")]
    items = [it for it in items if norm_text(it.get("label", "")) and len(norm_text(it["label"])) <= 24]
    pick_e = [items[i] for i in np.random.default_rng(a.seed).choice(len(items), size=min(a.estr_n, len(items)), replace=False)]
    for k, it in enumerate(pick_e):
        lit, dark = load_rgb(f"{a.estr_root}/{it['well_lit']}"), load_rgb(f"{a.estr_root}/{it['low_light']}")
        p_lit, p_dark = predict(rec, torch.stack([crops_estr(lit)]))[0], predict(rec, torch.stack([crops_estr(dark)]))[0]
        gt = norm_text(it["label"])
        ok = lambda p: norm_text(p) == gt
        estr_rows.append(dict(lit=lit, dark=dark, label=it["label"], p_lit=p_lit, p_dark=p_dark, ok_lit=ok(p_lit), ok_dark=ok(p_dark)))
        estr_rec.append(dict(id=it["id"], label=it["label"], pred_lit=p_lit, pred_dark=p_dark,
                             exact_lit=ok(p_lit), exact_dark=ok(p_dark),
                             cer_lit=edit_distance(norm_text(p_lit), gt) / max(len(gt), 1),
                             cer_dark=edit_distance(norm_text(p_dark), gt) / max(len(gt), 1)))
        print(f"[estr {k + 1}/{len(pick_e)}] {it['id']}: {it['label']!r} lit={p_lit!r} dark={p_dark!r}")

    ni, li, di = len(icdar_rec), sum(r["exact_lit"] for r in icdar_rec), sum(r["exact_dark"] for r in icdar_rec)
    ne, le, de = len(estr_rec), sum(r["exact_lit"] for r in estr_rec), sum(r["exact_dark"] for r in estr_rec)
    summary = dict(
        config=dict(model="microsoft/trocr-base-printed (frozen, no training)", route="oracle GT boxes + rectify",
                    crop_hw=list(crop_hw), darken="physical/dark online", n_icdar_images=len(pick), n_estr_pairs=len(pick_e), seed=a.seed),
        icdar=dict(images=len(pick), instances=ni, exact_lit=li, exact_dark=di,
                   exact_rate_lit=round(li / ni, 4), exact_rate_dark=round(di / ni, 4),
                   mean_cer_lit=round(float(np.mean([r["cer_lit"] for r in icdar_rec])), 4),
                   mean_cer_dark=round(float(np.mean([r["cer_dark"] for r in icdar_rec])), 4)),
        estr=dict(pairs=ne, exact_lit=le, exact_dark=de, exact_rate_lit=round(le / ne, 4), exact_rate_dark=round(de / ne, 4),
                  mean_cer_lit=round(float(np.mean([r["cer_lit"] for r in estr_rec])), 4),
                  mean_cer_dark=round(float(np.mean([r["cer_dark"] for r in estr_rec])), 4)),
        icdar_records=icdar_rec, estr_records=estr_rec)
    json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=2, ensure_ascii=False)

    # composites: ICDAR top by (any dark ok, any lit ok, n inst); ESTR prefer lit-ok/dark-bad
    ranked = sorted(range(len(icdar_bands)), key=lambda i: (icdar_meta[i]["ok_dark"], icdar_meta[i]["ok_lit"], icdar_meta[i]["n"]), reverse=True)
    sel = [icdar_bands[i] for i in ranked[:8]]
    W = max(b.shape[1] for b in sel) + 16
    H = sum(b.shape[0] for b in sel) + 40 + 10 * len(sel)
    canvas = np.full((H, W, 3), 255, np.uint8)
    put(canvas, "ICDAR2015 | oracle GT-box rectification + frozen pretrained TrOCR (no training) | box/text green=exact yellow=partial red=miss", (8, 24), BLUE, 0.55, 1)
    y = 40
    for b in sel:
        canvas[y : y + b.shape[0], 8 : 8 + b.shape[1]] = b
        y += b.shape[0] + 10
    cv2.imwrite(f"{a.out}/icdar2015_oracle.png", canvas)
    compose_estr(estr_rows, f"{a.out}/estr_dark_vs_lit.png")
    print(f"\n== ICDAR2015: {ni} instances, exact lit {li}/{ni} ({100 * li / ni:.0f}%), dark {di}/{ni} ({100 * di / ni:.0f}%)")
    print(f"== ESTR: {ne} pairs, exact lit {le}/{ne} ({100 * le / ne:.0f}%), dark {de}/{ne} ({100 * de / ne:.0f}%)")
    print(f"outputs -> {a.out}/icdar2015_oracle.png, {a.out}/estr_dark_vs_lit.png, {a.out}/summary.json")


if __name__ == "__main__":
    main()
