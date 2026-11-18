"""Evaluate explicitly selected checkpoints on the 500-image LATeD test set."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime.resolve()))
    from seeread.v5.seeing import build_detector
    from seeread.v5.seeing_decode import decode
    from seeread.metrics.spotting import SpottingEvaluator
    spec = importlib.util.spec_from_file_location('lated_test_metrics', ROOT / 'seeread/v5/lated_evaluation.py')
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    manifest = ROOT / 'data/annotations/v5/detection/latedspot_test.jsonl'
    protocol = json.loads((manifest.parent / 'provenance.json').read_text())
    assert sha256(manifest) == protocol['frozen_manifests']['latedspot_test']['sha256']
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if len(rows) != 500 or len({r['image'] for r in rows}) != 500:
        raise ValueError('Expected exactly the 500 unique official test images')
    fingerprints = {r['path']: r for line in (manifest.parent / 'image_fingerprints.jsonl').open()
                    if (r := json.loads(line)) and r.get('path')}
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if saved.get('resume_safe') is not True:
        raise ValueError('A healthy saved checkpoint is required')
    if args.checkpoint.name.startswith('test_candidate_epoch_'):
        expected_epoch = int(args.checkpoint.stem.rsplit('_', 1)[-1])
        if saved['state']['epoch'] != expected_epoch or saved['state']['position'] != 0:
            raise ValueError('Captured checkpoint does not match its named completed epoch')
    cfg = saved['config']
    detector_cfg = dict(cfg['detector'], pretrained=False)
    model = build_detector(detector_cfg)
    model.load_state_dict(saved['model'], strict=True)
    model.eval().to(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', test_used_for_selection=True, checkpoint=str(args.checkpoint),
                  checkpoint_sha256=sha256(args.checkpoint), checkpoint_state=saved['state'],
                  manifest_sha256=sha256(manifest), decoder=cfg['decoder'], runtime=str(args.runtime),
                  scope='official LATeD 500 images; local CTW-VOC protocol port, not verified author LATeD evaluator')
    (args.output / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    del saved
    totals, internal, start = Counter(), SpottingEvaluator(iou_thr=.5), time.monotonic()
    height, width = cfg['data']['image_hw']
    with (args.output / 'records.jsonl').open('x') as out, torch.inference_mode():
        for number, row in enumerate(rows, 1):
            path = ROOT / 'data' / row['image']
            fp = fingerprints.get(row['image'])
            if not fp or sha256(path) != fp['sha256']:
                raise ValueError(f'Image fingerprint mismatch: {row["image"]}')
            image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if image is None:
                raise FileNotFoundError(path)
            h, w = image.shape[:2]
            rgb = cv2.cvtColor(cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float().to(args.device) / 255
            pred = model(tensor)
            maps = {key: pred[key][0] for key in ('text', 'center', 'height', 'angle')}
            predictions = decode(maps, image_hw=(h, w), **cfg['decoder'])
            predictions = [dict(poly=x['poly'].tolist(), score=float(x['score']), text='') for x in predictions]
            gt = row['instances']
            counts, _ = metrics.image_counts(predictions, gt)
            totals.update(counts)
            internal.add(predictions, [dict(x, text='###') for x in gt])
            out.write(json.dumps(dict(image=row['image'], predictions=predictions, counts=counts)) + '\n')
            if number % 10 == 0:
                out.flush()
                print(json.dumps(dict(evaluated=number, seconds=time.monotonic()-start)), flush=True)
    report.update(status='complete', evaluated=len(rows), elapsed_seconds=time.monotonic()-start,
                  ctw_voc_port=metrics.summary(totals)['detection'], counts=dict(totals),
                  internal=internal.summary()['all'])
    (args.output / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['ctw_voc_port']), flush=True)


if __name__ == '__main__':
    main()
