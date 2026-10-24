"""Same-checkpoint enhancement removal on Reading development words only.

The joint recognizer is held fixed. These inference removals diagnose enhancer
use; they are not independently retrained module ablations or test selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from seeread.v5.enhancers import SelfDACEEnhancer
from seeread.v5.reading import WordCropDataset, sha256
from seeread.v5.recognition import build_pipeline
from tools.v5_detection_pilot import require_cuda
from tools.v5_reading import evaluate, source_hashes, validate_checkpoint, write_json


def run(checkpoint, output):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = saved['config']
    validate_checkpoint(saved, cfg)
    if saved.get('resume_safe') is not True or saved['source_hashes'] != source_hashes():
        raise ValueError('Use a healthy joint checkpoint from the current Reading implementation')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='initializing', checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                  checkpoint_step=saved['state']['global_step'], source_hashes=source_hashes(),
                  ablation_source_sha256=sha256(__file__), results={},
                  scope='dev-only inference removal; same joint OCR/enhancer weights, no retraining or test inference')
    write_json(output / 'run.json', report)
    try:
        data = cfg['data']
        dataset = WordCropDataset(data['dev_manifest'], root=data['root'], split='dev',
                                  input_hw=data['input_hw'], verify_hashes=True)
        if dataset.provenance['sha256'] != saved['manifests']['dev']['sha256']:
            raise ValueError('The development manifest differs from the selected checkpoint')
        device = require_cuda('cuda:0')
        fraction = float(cfg['train']['memory_fraction'])
        if not 0 < fraction <= .35:
            raise ValueError('Development ablation retains the GPU memory cap of0.35')
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        model = build_pipeline(cfg)
        model.load_state_dict(saved['model'], strict=True)
        del saved
        model.requires_grad_(False).eval().to(device)
        enhancer = model.enhancer
        if not isinstance(enhancer, SelfDACEEnhancer) or enhancer.denoise is None:
            raise ValueError('This ablation requires a genuine stage2 Self-DACE joint checkpoint')
        denoiser = enhancer.denoise
        report.update(status='running', dataset=dataset.provenance)
        write_json(output / 'run.json', report)
        try:
            for case in ('full', 'no_denoise', 'identity'):
                enhancer.denoise = None if case == 'no_denoise' else denoiser
                model.enhancer = nn.Identity() if case == 'identity' else enhancer
                result = evaluate(model, dataset, str(device), int(cfg['train']['batch_size']),
                                  output / (case + '.records.jsonl'))
                report['results'][case] = result
                write_json(output / 'run.json', report)
                print(json.dumps(dict(case=case, raw=result['raw'], complete=result['complete'])), flush=True)
        finally:
            enhancer.denoise = denoiser
            model.enhancer = enhancer
        report['status'] = 'complete'
        write_json(output / 'run.json', report)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output / 'run.json', report)
        raise
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.checkpoint, args.output)
