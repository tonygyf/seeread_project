"""Evaluate a healthy V5 checkpoint on a frozen development manifest.

Stage diagnostics reuse the trainer's preprocessing, decoder and evaluator;
they do not update parameters or replace formal benchmark evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from seeread.config import load_config
from seeread.v5.seeing import build_detector
from seeread.v5.seeing_dataset import SeeingDataset
from tools.v5_detection_pilot import evaluate, plain, require_cuda, runtime_versions, sha256, write_json
from tools.v5_train_detection import FORMAT, StageDatasetView, source_hashes, topology_signature


def run(checkpoint, config_path, output):
    cfg = plain(load_config(config_path))
    train, data = cfg['train'], cfg['data']
    fraction = float(train['memory_fraction'])
    if not 0 < fraction <= .35:
        raise ValueError('Development evaluation retains the GPU memory cap of0.35')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved.get('format') != FORMAT or saved.get('resume_safe') is not True:
        raise ValueError('Use a healthy V5 detection-stage checkpoint')
    if saved['config']['detector'] != cfg['detector']:
        raise ValueError('Evaluation detector configuration differs from the checkpoint')
    report = dict(status='initializing', checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                  checkpoint_stage=saved['config']['stage'], checkpoint_state=saved['state'],
                  config=cfg, config_path=str(config_path), config_sha256=sha256(config_path),
                  source_hashes={**source_hashes(), 'tools/v5_eval_detection.py': sha256(__file__)},
                  runtime=runtime_versions(),
                  scope='fixed development only; current-stage diagnostic, not final paper F1')
    write_json(output / 'run.json', report)
    started = time.monotonic()
    try:
        source = SeeingDataset(
            data['dev_manifest'], root=data['root'], split='dev',
            image_hw=data['image_hw'], feature_hw=data['feature_hw'], illumination='original',
            seed=int(train['seed']), center_shrink=float(data.get('center_shrink', .3)),
            index_cache_dir=data.get('index_cache_dir'), image_fingerprints=data.get('image_fingerprints'),
            verify_fingerprints=True, require_fingerprint=bool(data.get('require_fingerprint', False)),
            protocol_provenance=data.get('protocol_provenance'))
        device = require_cuda('cuda:0')
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        cv2.setNumThreads(1)
        model = build_detector(cfg['detector'])
        if topology_signature(model) != saved['topology']:
            raise ValueError('Checkpoint topology/source differs from the current detector')
        model.load_state_dict(saved['model'], strict=True)
        model.requires_grad_(False).eval().to(device)
        report.update(status='running', dataset=source.provenance,
                      training_checkpoint_provenance=saved['provenance'])
        del saved
        write_json(output / 'run.json', report)
        result = evaluate(model, StageDatasetView(source), device, int(train['val_batch_size']),
                          output / 'dev.records.jsonl', cfg['decoder'], float('inf'))
        report.update(status='complete' if result['complete'] else 'incomplete', result=result,
                      elapsed_seconds=time.monotonic() - started)
        write_json(output / 'run.json', report)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output / 'run.json', report)
        raise
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.checkpoint, args.config, args.output)
    print(json.dumps(result['result'], ensure_ascii=False, allow_nan=False), flush=True)
