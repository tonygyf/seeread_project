"""Interface-attribution experiment (READ-ONLY w.r.t. repo code; lives in tools/).

Question: how much of the GT-crop-oracle vs e2e gap comes from the det->rec crop interface?
Protocol (totaltext_test, deployment combo det=s3_joint_v2/best.pt + enh/rec=s2_rec_v2/last.pt):
  * per image: deterministic resize to cfg.data.size (eval geometry), det inference -> polygons;
  * greedy IoU>=0.5 matching vs GT (ignore GT may absorb preds, mirroring SpottingEvaluator);
  * for matched pairs, three rectify crops from the SAME image, all through enh+rec:
      A: GT-polygon centerline (training-time interface, upper reference)
      B: predicted centerline (current inference interface)
      C: predicted centerline extended by 0.4x local height along tangents (proposed fix)
  * two illumination conditions: lit (raw image) and dark (same darken() as the online
    synth pipeline, applied at ORIGINAL resolution before resize, as SeeReadDataset does).
Outputs: results/archive/diagnostics/attribution/summary.json + records_<cond>.jsonl + vis/*.png
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, cv2, torch
from shapely.geometry import Polygon

from seeread.config import load_config
from seeread.data.geometry import centerline_and_height
from seeread.data.lowlight_synth import darken
from seeread.metrics.spotting import norm_text, edit_distance, iou
from seeread.models.rectify import rectify_strip
from seeread.models.recognizer import build_recognizer
from seeread.models.seeread import SeeRead
from seeread.models.tsr import tsr_decode

ROOT, ANN = 'data', 'data/annotations/totaltext_test.jsonl'
DET_CKPT, REC_CKPT = 'results/archive/legacy/v2/s3_joint_v2/best.pt', 'results/archive/legacy/v2/s2_rec_v2/last.pt'
OUT = 'results/archive/diagnostics/attribution'
EXT = 0.4                      # extension length = EXT * local height (condition C)
SEED = 1234                    # darkening rng seed
CHUNK = 256
MAX_DET = 100

os.makedirs(OUT, exist_ok=True)
os.makedirs(f'{OUT}/vis', exist_ok=True)
cfg = load_config('configs/base.yaml', {'data': {}, 'train': {'amp': False}})
dev = 'cuda'
m = SeeRead(build_recognizer(cfg.rec), cfg).to(dev).eval()
sd_d = torch.load(DET_CKPT, map_location=dev)
sd_r = torch.load(REC_CKPT, map_location=dev)
m.det.load_state_dict(sd_d['det'])
m.enh.load_state_dict(sd_r['enh'])
m.rec.load_state_dict(sd_r['rec'])
HW = tuple(cfg.model.crop_hw)
SIZE, STRIDE = cfg.data.size, m.stride
print(f'loaded det={DET_CKPT} enh/rec={REC_CKPT} crop_hw={HW} rec={cfg.rec.type}', flush=True)


def _poly(p):
    q = Polygon(np.asarray(p, np.float64).reshape(-1, 2))
    return q if q.is_valid else q.buffer(0)


def read_rgb(p):
    im = cv2.imread(os.path.join(ROOT, p), cv2.IMREAD_COLOR)
    assert im is not None, p
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def to_canvas(img_u8, s, nh, nw):
    """Eval geometry: resize then top-left canvas, identical to SeeReadDataset(train=False)._apply."""
    img = cv2.resize(img_u8, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((SIZE, SIZE, 3), np.uint8)
    canvas[:min(nh, SIZE), :min(nw, SIZE)] = img[:SIZE, :SIZE]
    return canvas


def det_forward(x):
    with torch.no_grad():
        out = m.det(x)
        ctx = m.det.illum_context(x, out['c5'])
        tp = torch.sigmoid(out['text'][:, 0]).cpu().numpy()
        cp = torch.sigmoid(out['center'][:, 0]).cpu().numpy()
        geo = torch.cat([out['geo'][:, :1].clamp(min=0), out['geo'][:, 1:]], 1).cpu().numpy()
    dets = tsr_decode(tp[0], cp[0], geo[0], STRIDE, 0.5, 0.5)[:MAX_DET]
    for d in dets:                                   # same K<2 fix-up as SeeRead.infer
        if len(d['centers']) < 2:
            d['centers'] = np.concatenate([d['centers'] - [2, 0], d['centers'] + [2, 0]])
            d['heights'] = np.repeat(d['heights'], 2)
    return dets, ctx[0]


def extend_ends(centers, heights, ext=EXT):
    """Prepend/append one centerline point along the end tangents, 0.4x local height out."""
    c = np.asarray(centers, np.float32); h = np.asarray(heights, np.float32)
    t0 = c[1] - c[0]; t0 = t0 / max(np.linalg.norm(t0), 1e-6)
    t1 = c[-1] - c[-2]; t1 = t1 / max(np.linalg.norm(t1), 1e-6)
    c2 = np.concatenate([c[0:1] - t0[None] * ext * h[0], c, c[-1:] + t1[None] * ext * h[-1]], 0)
    h2 = np.concatenate([h[:1], h, h[-1:]], 0)
    return c2.astype(np.float32), h2.astype(np.float32)


rows = [json.loads(l) for l in open(ANN, encoding='utf-8') if l.strip()]
print(f'{len(rows)} test images', flush=True)

records = {'lit': [], 'dark': []}
n_valid_gt = 0
for idx, r in enumerate(rows):
    img = read_rgb(r['image'])
    H, W = img.shape[:2]
    s = SIZE / max(H, W)                              # train=False geometry
    nh, nw = int(round(H * s)), int(round(W * s))
    lit_u8 = to_canvas(img, s, nh, nw)
    rng = np.random.default_rng((SEED, idx))          # online synth darkening, dataset-style
    dark_o, meta = darken(img, rng, mode=cfg.data.synth_mode)
    dark_u8 = to_canvas(dark_o, s, nh, nw)
    conds = {'lit': dict(img=lit_u8, level='lit'),
             'dark': dict(img=dark_u8, level=meta['level'])}

    gts = []
    for gi, ins in enumerate(r['instances']):
        poly = np.asarray(ins['poly'], np.float32) * s          # canvas px
        text = ins.get('text', '')
        gts.append(dict(poly=poly, gp=_poly(poly), text=text,
                        ignore=bool(ins.get('ignore')) or norm_text(text) == '',
                        gi=gi))
    n_valid_gt += sum(1 for g in gts if not g['ignore'])

    for cond, cd in conds.items():
        x = torch.from_numpy(np.ascontiguousarray(cd['img'])).permute(2, 0, 1).float()[None].to(dev) / 255
        dets, ctx = det_forward(x)
        order = sorted(range(len(dets)), key=lambda i: -dets[i]['score'])
        used = set()
        matched = []
        for i in order:                                  # greedy IoU>=0.5, ignore GT may absorb (eval protocol)
            pp = _poly(dets[i]['poly'])
            best, bj = 0.0, -1
            for j, g in enumerate(gts):
                if j in used:
                    continue
                v = iou(pp, g['gp'])
                if v > best:
                    best, bj = v, j
            if best >= 0.5 and bj >= 0:
                used.add(bj)
                if not gts[bj]['ignore']:
                    matched.append((bj, i))
        cd['matched'] = matched
        # ---- build A/B/C crops for this condition on the fly
        buf = {'A': [], 'B': [], 'C': []}
        for bj, di in matched:
            g = gts[bj]; d = dets[di]
            c_gt, h_gt = centerline_and_height(g['poly'], k=7)
            crops = {
                'A': rectify_strip(x, torch.from_numpy(c_gt).to(dev), torch.from_numpy(h_gt).to(dev), HW),
                'B': rectify_strip(x, torch.from_numpy(d['centers']).to(dev), torch.from_numpy(d['heights']).to(dev), HW),
            }
            c_c, h_c = extend_ends(d['centers'], d['heights'], EXT)
            crops['C'] = rectify_strip(x, torch.from_numpy(c_c).to(dev), torch.from_numpy(h_c).to(dev), HW)
            for k in 'ABC':
                buf[k].append(dict(crop=crops[k][0].cpu(), ctx=ctx.cpu(), text=g['text'],
                                   img=idx, bj=bj, di=di))
        for k in 'ABC':
            for it in buf[k]:
                records[cond].append(dict(it, group=k))
    if (idx + 1) % 50 == 0:
        print(f'[{idx + 1}/{len(rows)}] lit_match={len(records["lit"]) // 3} dark_match={len(records["dark"]) // 3}', flush=True)


# ---------------------------------------------------------------- batch predict + metrics
def predict_group(items):
    """items: list of records with crop/ctx. Returns list of pred strings (enh+rec deployment path)."""
    preds = []
    with torch.no_grad():
        for i in range(0, len(items), CHUNK):
            cb = items[i:i + CHUNK]
            xb = torch.stack([it['crop'] for it in cb]).to(dev)
            zb = torch.stack([it['ctx'] for it in cb]).to(dev)
            y, _ = m.enh(xb, zb)
            preds += m.rec.predict(y)
            if (i // CHUNK) % 5 == 0:
                print(f'  predict {i + len(cb)}/{len(items)}', flush=True)
    return preds


summary = dict(config=dict(det=DET_CKPT, enh_rec=REC_CKPT, ann=ANN, n_images=len(rows),
                           crop_hw=list(HW), ext=EXT, darken=dict(mode=cfg.data.synth_mode, seed=SEED,
                                                                  level='online rng: mild/dark'),
                           protocol='A=GT centerline, B=pred centerline, C=pred centerline + 0.4h tangent extension; all crops enh+rec'),
               n_valid_gt=int(n_valid_gt), conditions={})
vis_pool = {}
for cond in ('lit', 'dark'):
    items = records[cond]
    by = {k: [it for it in items if it['group'] == k] for k in 'ABC'}
    gts_txt = [it['text'] for it in by['A']]
    for k in 'ABC':
        preds = predict_group(by[k])
        for it, p in zip(by[k], preds):
            it['pred'] = p
    per = []
    for a, b, c, gt in zip(by['A'], by['B'], by['C'], gts_txt):
        assert a['bj'] == b['bj'] == c['bj']
        ng = norm_text(gt)
        row = dict(img=a['img'], bj=a['bj'], di=a['di'], gt=gt, gt_norm=ng,
                   pred_A=a['pred'], pred_B=b['pred'], pred_C=c['pred'],
                   ok_A=norm_text(a['pred']) == ng, ok_B=norm_text(b['pred']) == ng, ok_C=norm_text(c['pred']) == ng,
                   ned_A=edit_distance(norm_text(a['pred']), ng) / max(len(ng), 1),
                   ned_B=edit_distance(norm_text(b['pred']), ng) / max(len(ng), 1),
                   ned_C=edit_distance(norm_text(c['pred']), ng) / max(len(ng), 1))
        per.append(row)
    res = dict(n_matched=len(per),
               det_match_rate=round(len(per) / max(n_valid_gt, 1), 4),
               word_acc={k: round(float(np.mean([r[f'ok_{k}'] for r in per])) if per else 0.0, 4) for k in 'ABC'},
               mean_ned={k: round(float(np.mean([r[f'ned_{k}'] for r in per])) if per else 0.0, 4) for k in 'ABC'},
               flip=dict(B_wrong_C_right=sum(1 for r in per if not r['ok_B'] and r['ok_C']),
                         B_right_C_wrong=sum(1 for r in per if r['ok_B'] and not r['ok_C']),
                         B_wrong_C_right_frac=round(float(np.mean([(not r['ok_B']) and r['ok_C'] for r in per])) if per else 0.0, 4),
                         B_right_C_wrong_frac=round(float(np.mean([r['ok_B'] and (not r['ok_C']) for r in per])) if per else 0.0, 4)))
    summary['conditions'][cond] = res
    vis_pool[cond] = per
    with open(f'{OUT}/records_{cond}.jsonl', 'w', encoding='utf-8') as f:
        for r in per:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'== {cond}: n_matched={res["n_matched"]} ({100 * res["det_match_rate"]:.1f}% of {n_valid_gt} valid GT) '
          f'acc A/B/C = {res["word_acc"]["A"]:.3f}/{res["word_acc"]["B"]:.3f}/{res["word_acc"]["C"]:.3f} '
          f'NED A/B/C = {res["mean_ned"]["A"]:.3f}/{res["mean_ned"]["B"]:.3f}/{res["mean_ned"]["C"]:.3f}', flush=True)

json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=2, ensure_ascii=False)

# ---------------------------------------------------------------- visualisation: A|B|C triplets
def sanitize(s, n=30):
    s = ''.join(ch if 32 <= ord(ch) < 127 else '?' for ch in str(s))
    return s[:n]


def put(img, txt, xy, color=(0, 0, 0), scale=0.45, th=1):
    cv2.putText(img, txt, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), th + 2, cv2.LINE_AA)
    cv2.putText(img, txt, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def make_triplet(row, crop_img_fn, tag):
    """row: per-instance record; crop_img_fn(kind)->HxWx3 uint8 strip for that instance."""
    SW, SH = 320, 80
    panel = np.full((SH + 96, SW * 3 + 32, 3), 255, np.uint8)
    for x_i, k in enumerate('ABC'):
        u8 = crop_img_fn(k)
        u8 = cv2.resize(u8, (SW, SH), interpolation=cv2.INTER_AREA)
        panel[4:4 + SH, 8 + x_i * (SW + 8):8 + x_i * (SW + 8) + SW] = u8
        put(panel, f'{k}: {sanitize(row[f"pred_{k}"])}', (8 + x_i * (SW + 8), SH + 24),
            (40, 160, 40) if row[f'ok_{k}'] else (0, 0, 220))
    put(panel, f'[{tag}] img#{row["img"]}  GT: {sanitize(row["gt"])}', (8, SH + 52), (0, 0, 0), 0.5, 1)
    put(panel, 'green=exact match', (8, SH + 76), (120, 120, 120), 0.42, 1)
    return panel


# re-materialise strips for the chosen cases: rerun rectify for a handful of (img, bj, di) pairs
def strips_for_case(cond, row):
    r = rows[row['img']]
    img = read_rgb(r['image'])
    H, W = img.shape[:2]
    s = SIZE / max(H, W)
    nh, nw = int(round(H * s)), int(round(W * s))
    base = img if cond == 'lit' else darken(img, np.random.default_rng((SEED, row['img'])), mode=cfg.data.synth_mode)[0]
    u8 = to_canvas(base, s, nh, nw)
    x = torch.from_numpy(np.ascontiguousarray(u8)).permute(2, 0, 1).float()[None].to(dev) / 255
    g = np.asarray(r['instances'][row['bj']]['poly'], np.float32) * s
    dets, _ = det_forward(x)
    d = dets[row['di']]
    c_gt, h_gt = centerline_and_height(g, k=7)
    c_c, h_c = extend_ends(d['centers'], d['heights'], EXT)
    outs = {}
    for k, (cc, hh) in dict(A=(c_gt, h_gt), B=(d['centers'], d['heights']), C=(c_c, h_c)).items():
        st = rectify_strip(x, torch.from_numpy(np.asarray(cc, np.float32)).to(dev),
                           torch.from_numpy(np.asarray(hh, np.float32)).to(dev), HW)[0].cpu().numpy()
        outs[k] = (np.clip(st.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)[:, :, ::-1]  # RGB->BGR
    return outs


saved = 0
for cond in ('dark', 'lit'):
    per = vis_pool[cond]
    prefer = [r for r in per if (not r['ok_B']) and r['ok_C']]           # C fixes B: truncation story
    fallback = [r for r in per if r['ok_B'] and not r['ok_C']]           # extension cost cases
    sel_ids = {id(r) for r in prefer + fallback}
    rest = [r for r in per if id(r) not in sel_ids]
    picks = (prefer + fallback + rest)[:3]
    for r in picks:
        try:
            outs = strips_for_case(cond, r)
        except Exception as e:
            print('vis skip', e, flush=True)
            continue
        panel = make_triplet(r, lambda k: outs[k], f'{cond} img#{r["img"]} gt#{r["bj"]} det#{r["di"]}')
        cv2.imwrite(f'{OUT}/vis/{cond}_case{saved:02d}_img{r["img"]}_gt{r["bj"]}.png', panel)
        saved += 1
print(f'vis: {saved} triplets -> {OUT}/vis/', flush=True)
print('DONE', flush=True)
