"""D1(+D2) per-matched-pair failure attribution for the det->rec interface (review docs/07 §3 D1/D2).
Classifier: tools/legacy/diag_geometry.py (numpy only). Protocol mirrors tools/legacy/attribution.py EXACTLY:
same ckpts, same eval resize/canvas, same online darkening rng seed, same greedy IoU>=0.5 matching,
so matched pairs are identical to results/archive/diagnostics/attribution/records_<cond>.jsonl (cross-checked by key).

Per matched pair we re-derive:
  gt_centers/gt_heights  from the GT polygon via centerline_and_height (k=7, same as A path);
  pred_centers/pred_heights from tsr_decode (after the K<2 fix-up, reading order as fed to rectify_strip);
  cause = classify(...)  in {flipped_order, height_wrong, center_drift, too_short, too_long, geometry_ok}.
D2 rot180 oracle: every A/B strip is ALSO recognised after a 180-degree rotation. If rot reads correct
where the original fails, the strip was upside-down (order flip) rather than blurry/wrong-geometry:
  - B_rot fixes B failures  -> predicted centerline order flipped (tsr_decode canonicalisation failed);
  - A_rot fixes A failures  -> GT centerline order flipped (split_polygon trusts the annotation's
    first-half traversal; TT polygons that do not start with an L->R top edge come out reversed).
Outputs: results/archive/diagnostics/attribution/d1_geometry.json + d1_records_<cond>.jsonl (+ printed summarize() tables).
Run:  docker compose run --rm -e CUDA_VISIBLE_DEVICES=0 seeread python tools/legacy/diag_d1_geometry.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, torch
from shapely.geometry import Polygon

from diag_geometry import classify, summarize

from seeread.config import load_config
from seeread.data.geometry import centerline_and_height
from seeread.data.lowlight_synth import darken
from seeread.metrics.spotting import norm_text, edit_distance, iou
from seeread.models.rectify import rectify_strip
from seeread.models.recognizer import build_recognizer
from seeread.models.seeread import SeeRead
from seeread.models.tsr import tsr_decode

ROOT, ANN = 'data', 'data/annotations/totaltext_test.jsonl'
DET_CKPT, REC_CKPT = 'results/archive/legacy/v2/s3_joint_v2/best.pt', 'results/archive/legacy/s4_rec_v4/best.pt'
OUT = 'results/archive/diagnostics/attribution'
SEED = 1234                    # must equal attribution.py so the dark rng reproduces
CHUNK = 256
MAX_DET = 100

os.makedirs(OUT, exist_ok=True)
cfg = load_config('configs/legacy/s4_rec_v4.yaml', {'data': {}, 'train': {'amp': False}})
dev = 'cuda'
m = SeeRead(build_recognizer(cfg.rec), cfg).to(dev).eval()
m.det.load_state_dict(torch.load(DET_CKPT, map_location=dev)['det'])
sd_r = torch.load(REC_CKPT, map_location=dev)
m.enh.load_state_dict(sd_r['enh']); m.rec.load_state_dict(sd_r['rec'])
HW = tuple(cfg.model.crop_hw)
SIZE, STRIDE = cfg.data.size, m.stride
print(f'loaded det={DET_CKPT} enh/rec={REC_CKPT} crop_hw={HW} rec={cfg.rec.type}', flush=True)

prev = {c: {(r['img'], r['bj'], r['di']): r for r in map(json.loads, open(f'{OUT}/records_{c}.jsonl', encoding='utf-8'))}
        for c in ('lit', 'dark')}


def _poly(p):
    q = Polygon(np.asarray(p, np.float64).reshape(-1, 2))
    return q if q.is_valid else q.buffer(0)


def read_rgb(p):
    im = cv2.imread(os.path.join(ROOT, p), cv2.IMREAD_COLOR)
    assert im is not None, p
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def to_canvas(img_u8, s, nh, nw):
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
    for d in dets:
        if len(d['centers']) < 2:
            d['centers'] = np.concatenate([d['centers'] - [2, 0], d['centers'] + [2, 0]])
            d['heights'] = np.repeat(d['heights'], 2)
    return dets, ctx[0]


def gt_order_reversed(poly):
    """True if split_polygon's first-half traversal runs against canonical reading direction
    (L->R for horizontal-dominant polys, top->bottom for vertical-dominant) -> GT centerline reversed."""
    p = np.asarray(poly, np.float32)
    if len(p) < 4:
        return None
    c = p - p.mean(0)
    ax = np.linalg.svd(c, full_matrices=False)[2][0]
    horiz = abs(ax[0]) >= abs(ax[1])
    try:
        cen, _ = centerline_and_height(p, 7)
    except Exception:
        return None
    d = cen[-1] - cen[0]
    return bool(d[0] < 0) if horiz else bool(d[1] < 0)


rows = [json.loads(l) for l in open(ANN, encoding='utf-8') if l.strip()]
print(f'{len(rows)} test images', flush=True)

recs = {'lit': [], 'dark': []}
audit = dict(n_valid_gt=0, gt_reversed=0, matched=0, matched_gt_reversed=0)
for idx, r in enumerate(rows):
    img = read_rgb(r['image'])
    H, W = img.shape[:2]
    s = SIZE / max(H, W)
    nh, nw = int(round(H * s)), int(round(W * s))
    lit_u8 = to_canvas(img, s, nh, nw)
    rng = np.random.default_rng((SEED, idx))
    dark_o, meta = darken(img, rng, mode=cfg.data.synth_mode)
    dark_u8 = to_canvas(dark_o, s, nh, nw)
    conds = {'lit': dict(img=lit_u8, level='lit'), 'dark': dict(img=dark_u8, level=meta['level'])}

    gts = []
    for gi, ins in enumerate(r['instances']):
        poly = np.asarray(ins['poly'], np.float32) * s
        text = ins.get('text', '')
        gts.append(dict(poly=poly, gp=_poly(poly), text=text,
                        ignore=bool(ins.get('ignore')) or norm_text(text) == '', gi=gi))
    for g in gts:                                     # GT start-vertex order audit (canvas px, both conds share it)
        if g['ignore']:
            continue
        audit['n_valid_gt'] += 1
        if gt_order_reversed(g['poly']):
            audit['gt_reversed'] += 1

    for cond, cd in conds.items():
        x = torch.from_numpy(np.ascontiguousarray(cd['img'])).permute(2, 0, 1).float()[None].to(dev) / 255
        dets, ctx = det_forward(x)
        order = sorted(range(len(dets)), key=lambda i: -dets[i]['score'])
        used = set()
        matched = []
        for i in order:
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
        for bj, di in matched:
            g = gts[bj]; d = dets[di]
            audit['matched'] += 1
            rev = gt_order_reversed(g['poly'])
            if rev:
                audit['matched_gt_reversed'] += 1
            c_gt, h_gt = centerline_and_height(g['poly'], k=7)
            crop = rectify_strip(x, torch.from_numpy(c_gt).to(dev), torch.from_numpy(h_gt).to(dev), HW)[0].cpu()
            cropb = rectify_strip(x, torch.from_numpy(d['centers']).to(dev), torch.from_numpy(d['heights']).to(dev), HW)[0].cpu()
            z = ctx.cpu()
            recs[cond].append(dict(img=idx, bj=bj, di=di, gt=g['text'],
                                   pc=d['centers'].tolist(), ph=d['heights'].tolist(),
                                   gc=c_gt.tolist(), gh=h_gt.tolist(),
                                   crop_a=crop, crop_b=cropb, rot_a=torch.flip(crop, [-2, -1]),
                                   rot_b=torch.flip(cropb, [-2, -1]), ctx=z,
                                   gt_rev=bool(rev) if rev is not None else False))
    if (idx + 1) % 50 == 0:
        print(f'[{idx + 1}/{len(rows)}] lit={len(recs["lit"])} dark={len(recs["dark"])}', flush=True)


def predict_items(items, key):
    preds = []
    with torch.no_grad():
        for i in range(0, len(items), CHUNK):
            cb = items[i:i + CHUNK]
            xb = torch.stack([it[key] for it in cb]).to(dev)
            zb = torch.stack([it['ctx'] for it in cb]).to(dev)
            y, _ = m.enh(xb, zb)
            preds += m.rec.predict(y)
    return preds


def ned(p, t):
    return edit_distance(norm_text(p), norm_text(t)) / max(len(norm_text(t)), 1)


summary = dict(config=dict(det=DET_CKPT, enh_rec=REC_CKPT, ann=ANN, crop_hw=list(HW), seed=SEED,
                           classifier='tools/legacy/diag_geometry.py classify(flip_ratio=0.8,...)',
                           protocol='pairs identical to attribution.py; A/B strips + 180deg rotations through enh+rec'),
               gt_order_audit=audit, conditions={})
for cond in ('lit', 'dark'):
    items = recs[cond]
    print(f'== {cond}: {len(items)} matched pairs; predicting 4 strips/pair (A,B,rotA,rotB)', flush=True)
    for key, tag in (('crop_a', 'pred_A'), ('rot_a', 'pred_A_rot'), ('crop_b', 'pred_B'), ('rot_b', 'pred_B_rot')):
        for it, p in zip(items, predict_items(items, key)):
            it[tag] = p
    per, cls = [], []
    mism = 0
    for it in items:
        ng = norm_text(it['gt'])
        ok_A = norm_text(it['pred_A']) == ng; ok_B = norm_text(it['pred_B']) == ng
        cause, st = classify(np.asarray(it['pc'], np.float32), np.asarray(it['ph'], np.float32),
                             np.asarray(it['gc'], np.float32), np.asarray(it['gh'], np.float32))
        prev_r = prev[cond].get((it['img'], it['bj'], it['di']))
        if prev_r is not None and (prev_r['ok_A'] != ok_A or prev_r['ok_B'] != ok_B):
            mism += 1
        per.append(dict(img=it['img'], bj=it['bj'], di=it['di'], gt=it['gt'],
                        pc=it['pc'], ph=it['ph'], gc=it['gc'], gh=it['gh'],
                        pred_A=it['pred_A'], pred_A_rot=it['pred_A_rot'], pred_B=it['pred_B'], pred_B_rot=it['pred_B_rot'],
                        ok_A=ok_A, ok_B=ok_B,
                        ok_A_rot=norm_text(it['pred_A_rot']) == ng, ok_B_rot=norm_text(it['pred_B_rot']) == ng,
                        gt_rev=it['gt_rev'], cause=cause, stats=st))
        cls.append((cause, st))
    with open(f'{OUT}/d1_records_{cond}.jsonl', 'w', encoding='utf-8') as f:
        for r in per:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'-- {cond} B-path cause attribution (share among {sum(1 for r in per if not r["ok_B"])} B-failures):')
    res_b = summarize(cls, [r['ok_B'] for r in per])
    print(f'-- {cond} A-path cause attribution (share among {sum(1 for r in per if not r["ok_A"])} A-failures):')
    res_a = summarize(cls, [r['ok_A'] for r in per])
    nf = [r for r in per if not r['ok_B']]; nfa = [r for r in per if not r['ok_A']]
    rot = dict(B_fail=len(nf), B_fail_fixed_by_rot180=sum(r['ok_B_rot'] for r in nf),
               B_acc=round(float(np.mean([r['ok_B'] for r in per])), 4),
               B_acc_either_orient=round(float(np.mean([r['ok_B'] or r['ok_B_rot'] for r in per])), 4),
               A_fail=len(nfa), A_fail_fixed_by_rot180=sum(r['ok_A_rot'] for r in nfa),
               A_acc=round(float(np.mean([r['ok_A'] for r in per])), 4),
               A_acc_either_orient=round(float(np.mean([r['ok_A'] or r['ok_A_rot'] for r in per])), 4),
               vs_prev_records_mismatch=mism,
               flipped_bucket_gt_rev=sum(1 for r in per if r['cause'].startswith('flipped') and r['gt_rev']),
               flipped_bucket_n=sum(1 for r in per if r['cause'].startswith('flipped')))
    print(f'-- {cond} rot180 oracle: {rot}', flush=True)
    summary['conditions'][cond] = dict(n=len(per), word_acc=dict(A=rot['A_acc'], B=rot['B_acc']),
                                       causes_B=res_b, causes_A=res_a, rot180=rot)

json.dump(summary, open(f'{OUT}/d1_geometry.json', 'w'), indent=2, ensure_ascii=False)
print('DONE ->', f'{OUT}/d1_geometry.json', flush=True)
