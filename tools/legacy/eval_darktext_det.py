"""Evaluate the ported DarkText TextNet detector (TextSnake mask path) for det_P/R/F.

Preprocessing matches upstream exactly: ResizeSquareM (direct stretch to size x size, NO aspect
preservation / padding) + ImageNet normalisation; ground-truth polygons are scaled per-axis
([size/W, size/H]) to the stretched frame. This is deliberately DIFFERENT from the diag_det.py
`to_canvas` geometry, because the checkpoint was trained with direct-stretch.

Run (source shadowed + checkpoint mounted read-only + GPU):
  docker compose run --rm -e CUDA_VISIBLE_DEVICES=0 \
      -v "$(pwd):/workspace" -v /home/gyf/DarkText-private:/darktext:ro seeread \
      python tools/legacy/eval_darktext_det.py --size 640 --out results/archive/diagnostics/darktext_det
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, cv2, torch

from seeread.metrics.spotting import SpottingEvaluator
from seeread.models.textgraph import TextNet, MaskTextDetector

CKPT = '/darktext/checkpoints/pretrained/pretrained/textgraph_vgg_2964.pth'
ROOT = 'data'
TEST_ANN = 'data/annotations/totaltext_test.jsonl'
TRAIN_ANN = 'data/annotations/totaltext_train_main.jsonl'
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
THRS = (0.3, 0.4, 0.5)


def load_detector(dev='cuda', ckpt=CKPT):
    model = TextNet(backbone='vgg', is_training=False).to(dev)
    missing, unexpected = model.load_model(ckpt)
    print(f'loaded {ckpt}\n  missing={missing}\n  unexpected={unexpected}', flush=True)
    return MaskTextDetector(model)


def read_rgb(p):
    im = cv2.imread(os.path.join(ROOT, p), cv2.IMREAD_COLOR)
    assert im is not None, p
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def run_pass(det, ann, size, dev, limit=0, tag=''):
    rows = [json.loads(l) for l in open(ann, encoding='utf-8') if l.strip()]
    if limit:
        rows = rows[:limit]
    images = []
    for idx, r in enumerate(rows):
        img = read_rgb(r['image'])
        H, W = img.shape[:2]
        sx, sy = size / W, size / H
        u8 = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(np.ascontiguousarray(u8)).permute(2, 0, 1).float()[None].to(dev) / 255.0
        x = (x - torch.from_numpy(MEAN).to(dev).view(1, 3, 1, 1)) / torch.from_numpy(STD).to(dev).view(1, 3, 1, 1)
        preds = det.detect(x)
        gts = [dict(poly=(np.asarray(ins['poly'], np.float32) * np.array([sx, sy], np.float32)),
                    ignore=bool(ins.get('ignore')))
               for ins in r['instances']]
        images.append(dict(preds=preds, gts=gts, img=r['image']))
        if (idx + 1) % 50 == 0:
            print(f'[{tag}] {idx + 1}/{len(rows)} n_pred={len(preds)}', flush=True)
    return images


def detection_metrics(images, thr):
    ev = SpottingEvaluator(iou_thr=thr)
    for im in images:
        ev.add(im['preds'], im['gts'])
    a = ev.summary()['all']
    return dict(iou_thr=thr, n_gt=int(a['n_gt']), n_pred=int(ev.c_all['n_pred']),
                det_P=round(a['det_P'], 4), det_R=round(a['det_R'], 4), det_F=round(a['det_F'], 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('--ann', default=TEST_ANN)
    ap.add_argument('--size', type=int, default=640)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='results/archive/diagnostics/darktext_det')
    ap.add_argument('--dev', default='cuda')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    det = load_detector(a.dev, a.ckpt)
    images = run_pass(det, a.ann, a.size, a.dev, a.limit, f'test@{a.size}')
    metrics = [detection_metrics(images, t) for t in THRS]
    summary = dict(config=dict(ckpt=a.ckpt, ann=a.ann, size=a.size, limit=a.limit,
                               preprocess='ResizeSquareM direct-stretch + ImageNet norm'),
                   metrics=metrics)
    json.dump(summary, open(f'{a.out}/det_f_size{a.size}.json', 'w'), indent=2, ensure_ascii=False)
    print('IoU scan:', metrics, flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
