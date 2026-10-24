"""Replay one saved training batch without optimizer updates or checkpoint edits."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from seeread.v5.seeing import build_detector
from seeread.v5.seeing_dataset import SeeingDataset
from seeread.v5.seeing_losses import detection_loss
from tools.v5_train_detection import StageDatasetView, collate_stage, source_hashes
from tools.v5_detection_pilot import require_cuda, restore_rng, sha256


def scalar(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def tensor_summary(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                finite=bool(torch.isfinite(value).all()),
                minimum=scalar(value.min()), maximum=scalar(value.max()))


def loss_summary(prediction, targets):
    losses = detection_loss(prediction, targets)
    return dict(total=scalar(sum(losses.values())),
                components={key: scalar(value) for key, value in losses.items()},
                dtypes={key: str(value.dtype) for key, value in losses.items()})


@torch.no_grad()
def diagnose(checkpoint_path, output, *, trace_features=False, include_bf16=False):
    if output.exists():
        raise FileExistsError(f'Preserving existing diagnostic: {output}')
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg, state = saved['config'], saved['state']
    data, train = cfg['data'], cfg['train']
    position, epoch = int(state['position']), int(state['epoch'])
    indices = saved['sampler']['order'][position:position + int(train['batch_size'])].tolist()
    source = SeeingDataset(
        data['train_manifest'], root=data['root'], split='train',
        image_hw=data['image_hw'], feature_hw=data['feature_hw'],
        augment=data.get('augment'), seed=int(train['seed']),
        center_shrink=data.get('center_shrink', .3),
        image_fingerprints=data.get('image_fingerprints'),
        require_fingerprint=data.get('require_fingerprint', False),
        protocol_provenance=data.get('protocol_provenance'))
    dataset = StageDatasetView(source)
    rows = [dataset[(epoch, int(index))] for index in indices]
    batch = collate_stage(rows)
    report = dict(status='running', checkpoint=str(checkpoint_path),
                  checkpoint_sha256=sha256(checkpoint_path),
                  diagnostic_only=True, checkpoint_resume_safe=saved['resume_safe'],
                  step=state['step'], epoch=epoch, position=position, indices=indices,
                  images=[row['row']['image'] for row in rows],
                  input=tensor_summary(batch['image']),
                  targets={key: tensor_summary(value) for key, value in batch['targets'].items()},
                  source_hashes=source_hashes(),
                  checkpoint_source_hashes=saved['provenance']['source_hashes'],
                  passes={})
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        device = require_cuda('cuda:0')
        torch.cuda.set_per_process_memory_fraction(float(train['memory_fraction']), 0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        model = build_detector(cfg['detector'])
        model.load_state_dict(saved['model'], strict=True)
        model.to(device).train()
        buffers = {key: value.detach().clone() for key, value in model.named_buffers()}
        batch['image'], batch['mask'] = batch['image'].to(device), batch['mask'].to(device)
        batch['targets'] = {key: value.to(device) for key, value in batch['targets'].items()}
        passes = [('amp', True, torch.float16)]
        if include_bf16:
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError('The explicit BF16 probe requires native BF16 support')
            passes.append(('bf16', True, torch.bfloat16))
        passes.append(('fp32', False, torch.float32))
        for name, amp, dtype in passes:
            for key, value in model.named_buffers():
                value.copy_(buffers[key])
            restore_rng(saved['rng'])
            center_trace = {}
            head_traces = {}
            feature_trace = {}
            handles = []
            if trace_features:
                for key, layer in model.named_modules():
                    if (key.startswith(('pm.1', 'pm.2')) and
                            isinstance(layer, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.GroupNorm))):
                        handles.append(layer.register_forward_hook(
                            lambda module, inputs, value, key=key: feature_trace.__setitem__(
                                key, dict(input=tensor_summary(inputs[0]), output=tensor_summary(value)))))
            for head in ('text', 'center', 'geometry'):
                trace = head_traces[head] = {}
                if head == 'center':
                    center_trace = trace
                for key, layer in getattr(model, head + '_head').named_modules():
                    if isinstance(layer, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU)):
                        handles.append(layer.register_forward_hook(
                            lambda module, inputs, value, key=key, trace=trace: trace.__setitem__(
                                key, dict(input=tensor_summary(inputs[0]), output=tensor_summary(value)))))
            with torch.autocast('cuda', enabled=amp, dtype=dtype):
                prediction = model(batch['image'], with_scm=True, mask=batch['mask'])
                original = loss_summary(prediction, batch['targets'])
            for handle in handles:
                handle.remove()
            result = dict(outputs={key: tensor_summary(value) for key, value in prediction.items()
                                   if isinstance(value, torch.Tensor)}, loss=original,
                          center_head_trace=center_trace, head_traces=head_traces,
                          feature_trace=feature_trace)
            if amp:
                with torch.autocast('cuda', enabled=False):
                    converted = {key: value.float() if isinstance(value, torch.Tensor) and key != 'feat' else value
                                 for key, value in prediction.items()}
                    result['same_outputs_fp32_loss'] = loss_summary(converted, batch['targets'])
                    raw = (prediction['sem_main'].float() - prediction['scm_sem'].float()).square().mean(1)
                    result['semantic_error_fp32'] = dict(tensor_summary(raw),
                                                        mean=scalar(raw.mean()), sum=scalar(raw.sum()))
                del converted
            report['passes'][name] = result
            print(json.dumps(dict(pass_name=name, result=result), allow_nan=False), flush=True)
            del prediction
        report.update(status='complete', peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        with output.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trace-features', action='store_true')
    parser.add_argument('--include-bf16', action='store_true')
    args = parser.parse_args()
    diagnose(args.checkpoint, args.output, trace_features=args.trace_features, include_bf16=args.include_bf16)
