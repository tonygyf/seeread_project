"""Separate connector-domain LoRA adaptation, reusing the unchanged baseline trainer."""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seeread.v5.reading import sha256
from seeread.v5.recognition import build_pipeline
from tools.v5_reading import FORMAT, run, source_hashes


def adaptation_config(saved, parent, manifest_dir, output):
    if saved.get('format') != FORMAT or saved.get('resume_safe') is not True:
        raise ValueError('Adaptation requires a healthy independent Reading checkpoint')
    if saved['source_hashes'] != source_hashes():
        raise ValueError('Parent Reading implementation differs from the current baseline')
    cfg = copy.deepcopy(saved['config'])
    cfg['experiment'] = dict(id='C_read_lora', role='fusion crop adaptation; not independent paper baseline',
                             parent=str(parent), parent_sha256=sha256(parent),
                             adapter_source_sha256=sha256(__file__),
                             data_protocol_sha256=sha256(manifest_dir / 'protocol.json'))
    cfg['enhancer']['freeze'] = True
    cfg['recognizer']['trainability'] = 'lora'
    cfg['data'].update(train_manifest=str(manifest_dir / 'train.jsonl'),
                       dev_manifest=str(manifest_dir / 'dev.jsonl'))
    cfg['train'].update(out_dir=str(output), max_steps=1200, max_seconds=1200,
                        eval_every=200, eval_initial=True, lr=2e-5)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=Path('results/v5/reading/R4str_joint/best.pt'))
    parser.add_argument('--manifests', type=Path, default=Path('data/annotations/v5/connector_reading_20261025'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--max-steps', type=int, default=1200)
    parser.add_argument('--max-seconds', type=float, default=1200)
    args = parser.parse_args()
    saved = torch.load(args.parent, map_location='cpu', weights_only=False)
    cfg = adaptation_config(saved, args.parent, args.manifests, args.output)
    cfg['train'].update(max_steps=args.max_steps, max_seconds=args.max_seconds)

    def factory(config):
        model = build_pipeline(config)
        model.load_state_dict(saved['model'], strict=True)
        names = [name for name, p in model.named_parameters() if p.requires_grad]
        if not names or any('lora_' not in name for name in names):
            raise ValueError('Only recognizer LoRA may train in this controlled adaptation')
        model.recognizer.provenance = dict(model.recognizer.provenance,
            adaptation_parent=cfg['experiment']['parent_sha256'],
            initialization='independent R4 best; new optimizer/sampler; fixed enhancer')
        return model

    report = run(cfg, 'train', 'cuda:0', pipeline_factory=factory, resume=args.resume)
    print(json.dumps(dict(status=report['status'], state=report['state'])), flush=True)


if __name__ == '__main__':
    main()
