"""D3/D4/D6/D7 detector-only diagnostics (docs/07 §3). Zero-training, no recognizer.

Detector = frozen v2 det (results/archive/legacy/v2/s3_joint_v2/best.pt), same forward as tools/legacy/attribution.py:
eval resize (long side -> size), top-left canvas, tsr_decode stride=4 (incl. v4 _trim_ends).

  D3  detection IoU threshold scan {0.3,0.4,0.5} on TT test @640  -> "polygon geometry off" vs "true miss"
  D4  det_F on TT train_main @640                                -> under-train/capacity vs overfit/domain gap
  D6  fragmentation/merge stats on TT test @640                  -> explain det_P
  D7  high-res det_F @1200 (TT test)                             -> resolution contribution to detect recall

Run (source shadowed so new tools/ are visible; GPU via -e CUDA_VISIBLE_DEVICES):
  docker compose run --rm -v "$(pwd):/workspace" -e CUDA_VISIBLE_DEVICES=0 seeread \
      python tools/legacy/diag_det.py --tasks d3,d6,d7 --limit 20 --out results/archive/diagnostics/diag_det
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, cv2, torch
from shapely.geometry import Polygon

from seeread.config import load_config
from seeread.metrics.spotting import SpottingEvaluator, iou, _poly
from seeread.models.detector import SeeReadDetector
from seeread.models.tsr import tsr_decode

DET_CKPT = 'results/archive/legacy/v2/s3_joint_v2/best.pt'
ROOT = 'data'
TEST_ANN = 'data/annotations/totaltext_test.jsonl'
TRAIN_ANN = 'data/annotations/totaltext_train_main.jsonl'
STRIDE = 4
MAX_DET = 200
THRS = (0.3, 0.4, 0.5)


def load_detector(dev='cuda'):
    cfg = load_config('configs/base.yaml', {})
    det = SeeReadDetector(ch=cfg.model.det_ch, pretrained=False, ctx_dim=cfg.model.ctx_dim).to(dev).eval()
    det.load_state_dict(torch.load(DET_CKPT, map_location=dev)['det'])
    return det


def read_rgb(p):
    im = cv2.imread(os.path.join(ROOT, p), cv2.IMREAD_COLOR)
    assert im is not None, p
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def to_canvas(img_u8, s, nh, nw, size):
    img = cv2.resize(img_u8, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:min(nh, size), :min(nw, size)] = img[:size, :size]
    return canvas


def det_forward(det, x):
    with torch.no_grad():
        out = det(x)
        tp = torch.sigmoid(out['text'][:, 0]).cpu().numpy()
        cp = torch.sigmoid(out['center'][:, 0]).cpu().numpy()
        geo = torch.cat([out['geo'][:, :1].clamp(min=0), out['geo'][:, 1:]], 1).cpu().numpy()
    return tsr_decode(tp[0], cp[0], geo[0], STRIDE, 0.5, 0.5)[:MAX_DET]


def run_pass(det, ann, size, dev, limit=0, tag=''):
    rows = [json.loads(l) for l in open(ann, encoding='utf-8') if l.strip()]
    if limit:
        rows = rows[:limit]
    images = []
    for idx, r in enumerate(rows):
        img = read_rgb(r['image'])
        H, W = img.shape[:2]
        s = size / max(H, W)
        nh, nw = int(round(H * s)), int(round(W * s))
        u8 = to_canvas(img, s, nh, nw, size)
        x = torch.from_numpy(np.ascontiguousarray(u8)).permute(2, 0, 1).float()[None].to(dev) / 255
        dets = det_forward(det, x)
        preds = [dict(poly=np.asarray(d['poly'], np.float32), score=d['score']) for d in dets]
        gts = [dict(poly=(np.asarray(ins['poly'], np.float32) * s), ignore=bool(ins.get('ignore')))
               for ins in r['instances']]
        images.append(dict(preds=preds, gts=gts, img=r['image']))
        if (idx + 1) % 50 == 0:
            print(f'[{tag}] {idx + 1}/{len(rows)}', flush=True)
    return images


def detection_metrics(images, thr):
    ev = SpottingEvaluator(iou_thr=thr)
    for im in images:
        ev.add(im['preds'], im['gts'])
    a = ev.summary()['all']
    return dict(iou_thr=thr, n_gt=int(a['n_gt']), n_pred=int(ev.c_all['n_pred']),
                det_P=round(a['det_P'], 4), det_R=round(a['det_R'], 4), det_F=round(a['det_F'], 4))


def greedy_labels(preds, gts, thr):
    """Per-pred label in {'tp','fp','absorbed'}, mirroring SpottingEvaluator (score-desc greedy,
    ignore/don't-care GT absorbs its matched pred)."""
    gp = [_poly(g['poly']) for g in gts]
    dont = [bool(g.get('ignore')) for g in gts]
    order = sorted(range(len(preds)), key=lambda i: -float(preds[i].get('score', 1.0)))
    used, label = set(), ['fp'] * len(preds)
    for i in order:
        pp = _poly(preds[i]['poly'])
        best, bj = 0.0, -1
        for j in range(len(gp)):
            if j in used:
                continue
            v = iou(pp, gp[j])
            if v > best:
                best, bj = v, j
        if best >= thr and bj >= 0:
            used.add(bj)
            label[i] = 'absorbed' if dont[bj] else 'tp'
    return label


def fragmentation(images, thr=0.5, thr_lo=0.3):
    agg = dict(n_pred=0, n_gt=0, tp=0, fp=0, absorbed=0, fp_touch=0, fp_spurious=0,
               merge_preds=0, gt_pred0=0, gt_pred1=0, gt_predge2=0, gt_merged_into=0)
    for im in images:
        preds, gts = im['preds'], im['gts']
        valid = [g for g in gts if not g['ignore']]
        P = [_poly(p['poly']) for p in preds]
        Gv = [_poly(g['poly']) for g in valid]
        agg['n_pred'] += len(preds)
        agg['n_gt'] += len(valid)
        label = greedy_labels(preds, gts, thr)
        agg['tp'] += label.count('tp')
        agg['fp'] += label.count('fp')
        agg['absorbed'] += label.count('absorbed')
        gt_touch = np.zeros(len(Gv), np.int32)               # preds (IoU>=thr_lo) touching each valid GT
        pred_touch = np.zeros(len(P), bool)                  # pred touches >=1 valid GT
        for i, pi in enumerate(P):
            touched, ngt = False, 0
            for j, gj in enumerate(Gv):
                v = iou(pi, gj)
                if v >= thr_lo:
                    gt_touch[j] += 1
                    touched, ngt = True, ngt + 1
            pred_touch[i] = touched
            agg['merge_preds'] += int(ngt >= 2)
        # classify FPs
        for i, l in enumerate(label):
            if l == 'fp':
                if pred_touch[i]:
                    agg['fp_touch'] += 1
                else:
                    agg['fp_spurious'] += 1
        agg['gt_pred0'] += int((gt_touch == 0).sum())
        agg['gt_pred1'] += int((gt_touch == 1).sum())
        agg['gt_predge2'] += int((gt_touch >= 2).sum())
    n_pred = max(agg['n_pred'], 1)
    n_gt = max(agg['n_gt'], 1)
    return dict(iou_thr=thr, touch_iou=thr_lo,
                det_P=round(agg['tp'] / max(agg['tp'] + agg['fp'], 1), 4),
                det_R=round(agg['tp'] / n_gt, 4),
                det_F=round(2 * (agg['tp'] / max(agg['tp'] + agg['fp'], 1)) * (agg['tp'] / n_gt)
                            / max((agg['tp'] / max(agg['tp'] + agg['fp'], 1)) + (agg['tp'] / n_gt), 1e-9), 4),
                n_pred=agg['n_pred'], n_gt=agg['n_gt'], tp=agg['tp'], fp=agg['fp'], absorbed=agg['absorbed'],
                fp_touch_gt_ge03=agg['fp_touch'], fp_pure_spurious=agg['fp_spurious'],
                fp_touch_frac=round(agg['fp_touch'] / max(agg['fp'], 1), 4),
                fp_spurious_frac=round(agg['fp_spurious'] / max(agg['fp'], 1), 4),
                merge_preds=agg['merge_preds'], merge_pred_frac=round(agg['merge_preds'] / n_pred, 4),
                gt_with_0_preds=agg['gt_pred0'], gt_with_1_pred=agg['gt_pred1'], gt_with_ge2_preds=agg['gt_predge2'],
                gt_split_frac=round(agg['gt_predge2'] / n_gt, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', default='d3,d6,d7,d4', help='comma list of d3,d4,d6,d7')
    ap.add_argument('--limit', type=int, default=0, help='cap images per pass (smoke test)')
    ap.add_argument('--out', default='results/archive/diagnostics/diag_det')
    ap.add_argument('--dev', default='cuda')
    a = ap.parse_args()
    tasks = set(x.strip() for x in a.tasks.split(',') if x.strip())
    os.makedirs(a.out, exist_ok=True)
    det = load_detector(a.dev)
    cfg = dict(det=DET_CKPT, stride=STRIDE, limit=a.limit,
               test_ann=TEST_ANN, train_ann=TRAIN_ANN)
    summary = dict(config=cfg, tasks=sorted(tasks))

    # test-side passes (D3/D6 share the @640 pass)
    test640 = test1200 = train640 = None
    if tasks & {'d3', 'd6'}:
        test640 = run_pass(det, TEST_ANN, 640, a.dev, a.limit, 'test@640')
    if 'd7' in tasks:
        test1200 = run_pass(det, TEST_ANN, 1200, a.dev, a.limit, 'test@1200')
    if 'd4' in tasks:
        train640 = run_pass(det, TRAIN_ANN, 640, a.dev, a.limit, 'train@640')

    if test640 is not None:
        if 'd3' in tasks:
            summary['d3'] = [detection_metrics(test640, t) for t in THRS]
            json.dump(summary['d3'], open(f'{a.out}/d3_iou_scan.json', 'w'), indent=2)
            print('D3 IoU scan:', summary['d3'], flush=True)
        if 'd6' in tasks:
            summary['d6'] = fragmentation(test640)
            json.dump(summary['d6'], open(f'{a.out}/d6_fragmentation.json', 'w'), indent=2)
            print('D6 fragmentation:', json.dumps(summary['d6'], indent=2), flush=True)
    if test1200 is not None:
        summary['d7'] = dict(
            low_res_640=[detection_metrics(test640, 0.5)] if test640 is not None else [],
            high_res_1200=[detection_metrics(test1200, t) for t in THRS])
        json.dump(summary['d7'], open(f'{a.out}/d7_highres.json', 'w'), indent=2)
        print('D7 high-res:', json.dumps(summary['d7'], indent=2), flush=True)
    if train640 is not None:
        summary['d4'] = dict(train=[detection_metrics(train640, t) for t in THRS])
        json.dump(summary['d4'], open(f'{a.out}/d4_train_detf.json', 'w'), indent=2)
        print('D4 train det:', json.dumps(summary['d4'], indent=2), flush=True)

    json.dump(summary, open(f'{a.out}/summary.json', 'w'), indent=2, ensure_ascii=False)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()