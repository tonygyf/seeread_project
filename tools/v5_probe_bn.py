"""Swap detector BatchNorm running statistics on frozen development images."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def with_bn_stats(model, weights, statistics):
    """Copy only BN running buffers; preserve all learned weights and affine parameters."""
    from torch.nn.modules.batchnorm import _BatchNorm
    mixed = dict(weights)
    replaced = []
    for name, layer in model.named_modules():
        if not isinstance(layer, _BatchNorm) or not layer.track_running_stats:
            continue
        for suffix in ('running_mean', 'running_var', 'num_batches_tracked'):
            key = f'{name}.{suffix}' if name else suffix
            if weights[key].shape != statistics[key].shape:
                raise ValueError(f'Incompatible BatchNorm buffer: {key}')
            mixed[key] = statistics[key]
            replaced.append(key)
    return mixed, replaced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--lated', type=Path, required=True)
    parser.add_argument('--mlt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from tools.v5_lated import bootstrap
    bootstrap(args.runtime)
    import torch
    import numpy as np
    from torch.nn import functional as F
    from seeread.config import load_config
    from seeread.v5.seeing import build_detector
    from seeread.v5.seeing_decode import decode
    from seeread.v5.lated_annotations import read_jsonl
    from seeread.v5.lated_evaluation import image_counts, summary
    from seeread.v5.reading import sha256
    from tools.v5_eval_spotting import rgb_tensor, verified_rows
    from tools.v5_detection_pilot import plain

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Use a single authorized idle GPU container')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.35)
    config = plain(load_config('configs/v5/fusion_placement_dev.yaml'))
    datasets = verified_rows(config)
    datasets['lated'] = read_jsonl('data/annotations/v5/detection/latedspot_val.jsonl')
    checkpoints = {}
    for key, path in [('LATeD', args.lated), ('MLT', args.mlt)]:
        saved = torch.load(path, map_location='cpu', weights_only=False)
        checkpoints[key] = dict(model=saved['model'], config=saved['config'], state=saved['state'], sha256=sha256(path))
        del saved
    model = build_detector(checkpoints['LATeD']['config']['detector']).requires_grad_(False).eval().cuda()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', scope='development only; BN state mixing is a diagnostic intervention, not trained weights',
                  checkpoints={k:dict(sha256=v['sha256'],step=v['state']['step']) for k,v in checkpoints.items()},
                  decoder=config['decoder'], image_hw=config['detector_hw'], results={})
    for weight_name, bn_name in [('LATeD','LATeD'),('LATeD','MLT'),('MLT','MLT'),('MLT','LATeD')]:
        mixed, replaced = with_bn_stats(model, checkpoints[weight_name]['model'], checkpoints[bn_name]['model'])
        model.load_state_dict(mixed, strict=True)
        label = f'{weight_name}_weights_{bn_name}_BN'
        result = dict(bn_buffers=len(replaced), datasets={})
        for dataset_name, rows in datasets.items():
            total, records, foreground = Counter(), [], Counter()
            for row in rows:
                image = rgb_tensor(Path(config['data_root']) / row['image'], 'cuda:0')
                with torch.inference_mode():
                    maps = model(F.interpolate(image[None], size=config['detector_hw'], mode='bilinear', align_corners=False))
                predictions = decode(maps, image_hw=image.shape[-2:], **config['decoder'])
                counts, _ = image_counts(predictions, row['instances'])
                total.update(counts); records.append(dict(image=row['image'],**counts))
                for k in ('text','center'):
                    foreground[k] += float((maps[k].softmax(1)[:,1] >= .5).float().mean())
            result['datasets'][dataset_name] = dict(images=len(rows), **summary(total)['detection'],
                    counts=dict(total), mean_foreground={k:v/len(rows) for k,v in foreground.items()}, per_image=records)
            print(json.dumps(dict(variant=label,dataset=dataset_name,**{k:v for k,v in result['datasets'][dataset_name].items() if k!='per_image'})),flush=True)
        report['results'][label] = result
        (out/'run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    report['status'] = 'complete'
    (out/'run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    main()
