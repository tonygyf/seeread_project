"""Frozen-development detection diagnosis: decoder candidates and GT-map interventions.

GT interventions diagnose learned maps and target reconstruction; they are not
inference methods and never select a deployable decoder.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime.resolve()))
    import numpy as np
    import torch
    from torch.nn import functional as F
    from seeread.config import load_config
    from seeread.metrics.spotting import SpottingEvaluator
    from seeread.v5.seeing import build_detector
    from seeread.v5.seeing_data import targets_from_polygons
    from tools.v5_detection_pilot import plain, require_cuda, sha256, write_json
    from tools.v5_eval_spotting import rgb_tensor, verified_rows
    from tools.v5_train_detection import topology_signature

    spec = importlib.util.spec_from_file_location('candidate_decoder', ROOT / 'seeread/v5/seeing_decode.py')
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='initializing', checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
                  decoder_source_sha256=sha256(ROOT / 'seeread/v5/seeing_decode.py'),
                  script_sha256=sha256(__file__), runtime=str(args.runtime),
                  scope='fixed development only; GT interventions are diagnostics, not deployable scores', results={})
    write_json(args.output / 'run.json', report)
    try:
        device = require_cuda('cuda:0')
        torch.cuda.set_per_process_memory_fraction(.35, 0)
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if saved.get('resume_safe') is not True or saved['config']['stage']['name'] != 'target':
            raise ValueError('Require a healthy target checkpoint')
        model = build_detector(saved['config']['detector'])
        if topology_signature(model) != saved['topology']:
            raise ValueError('Runtime does not match checkpoint topology')
        model.load_state_dict(saved['model'], strict=True)
        model.requires_grad_(False).eval().to(device)
        config = plain(load_config('configs/v5/fusion_placement_dev.yaml'))
        datasets = verified_rows(config)
        dev = Path(saved['config']['data']['dev_manifest'])
        datasets['lated_dev'] = [json.loads(line) for line in dev.read_text().splitlines() if line.strip()]
        report.update(status='running', detector_step=saved['state']['step'],
                      manifests={**{k: sha256(v) for k, v in config['manifests'].items()}, 'lated_dev': sha256(dev)},
                      decoder=config['decoder'], results={})
        del saved
        write_json(args.output / 'run.json', report)
        started = time.monotonic()
        for domain, rows in datasets.items():
            evaluators = {}
            counts = {}
            with (args.output / f'{domain}.records.jsonl').open('x') as stream:
                for index, row in enumerate(rows):
                    image = rgb_tensor(Path(config['data_root']) / row['image'], device)
                    with torch.inference_mode():
                        output = model(F.interpolate(image[None], size=config['detector_hw'], mode='bilinear', align_corners=False))
                    pred = {k: output[k][0].float().cpu().numpy() for k in ('text', 'center', 'height', 'angle')}
                    target = targets_from_polygons(row['instances'], image.shape[-2:], pred['height'].shape[-2:])
                    logits = lambda mask: np.stack([np.where(mask, -12., 12.), np.where(mask, 12., -12.)]).astype(np.float32)
                    gt = dict(text=logits(target['text']), center=logits(target['center']),
                              height=target['height'][None], angle=target['angle'][None])
                    variants = dict(original=(pred, {}), all_contours=(pred, {'keep_all_contours': True}),
                                    text_03=(pred, {'text_threshold': .3}), center_03=(pred, {'center_threshold': .3}),
                                    both_03=(pred, {'text_threshold': .3, 'center_threshold': .3}),
                                    oracle_classification=({**pred, 'text': gt['text'], 'center': gt['center']}, {}),
                                    oracle_geometry=({**pred, 'height': gt['height'], 'angle': gt['angle']}, {}),
                                    oracle_all=(gt, {}))
                    for name, (maps, changes) in variants.items():
                        predictions = candidate.decode(maps, image_hw=image.shape[-2:], **{**config['decoder'], **changes})
                        evaluators.setdefault(name, SpottingEvaluator()).add([dict(p, text='') for p in predictions], row['instances'])
                        counts[name] = counts.get(name, 0) + len(predictions)
                        stream.write(json.dumps(dict(image=row['image'], variant=name,
                            predictions=[dict(p, poly=p['poly'].tolist()) for p in predictions]), allow_nan=False)+'\n')
                    if (index + 1) % 20 == 0:
                        print(json.dumps(dict(domain=domain, images=index+1, elapsed=time.monotonic()-started)), flush=True)
            report['results'][domain] = {k: dict(metrics=e.summary()['all'], raw_predictions=counts[k]) for k, e in evaluators.items()}
            write_json(args.output / 'run.json', report)
        report.update(status='complete', elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(args.output / 'run.json', report)


if __name__ == '__main__':
    main()
