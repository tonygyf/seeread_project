"""Freeze a baseline protocol, then evaluate official LSTR test exactly once.

ESTR test_real is deliberately unsupported until the final project method is
locked. Development training/evaluation remains in tools/v5_reading.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from seeread.config import load_config
from seeread.v5.reading import WordCropDataset, sha256
from seeread.v5.recognition import build_pipeline
from seeread.v5.enhancers import build_enhancer, SELF_DACE_REVISION
from tools.v5_reading import plain, source_hashes, runtime_versions, provenance, evaluate, validate_checkpoint, write_json

DATA_REVISION = '619e3c1d3bb27135da785b77d3c414a946e37000'
TEST_MANIFEST_SHA256 = '5e9ee412513b0d9ac17185550e04df1e8013d4242c3ce6c2aca3739aa8218197'
DATA_PROVENANCE = Path('data/annotations/v5/reading/provenance.json')
FORMAT = 'seeread_v5_locked_lstr_test_v1'


class LockedLSTRTest(WordCropDataset):
    allowed_splits = ('test',)

    def __init__(self, manifest, expected_sha256, **kwargs):
        if expected_sha256 != TEST_MANIFEST_SHA256:
            raise ValueError('LSTR test must match the manifest produced by the verified V5 preparation')
        if sha256(manifest) != expected_sha256:
            raise ValueError('Official test manifest changed after the protocol was locked')
        super().__init__(manifest, split='test', **kwargs)
        if len(self) != 6588:
            raise ValueError('Expected the complete 6588-word official LSTR test')
        for row in self.rows:
            d = row['dataset']
            if (d.get('repo') != 'lumimusta/Low-light_Scene_Text_Dataset' or d.get('revision') != DATA_REVISION or
                    d.get('original_split') != 'test' or row['id'] != 'test:' + str(d.get('row_id'))):
                raise ValueError('The test source is not the pinned official Reading release')
        if {r['dataset']['row_index'] for r in self.rows} != set(range(6588)):
            raise ValueError('Official LSTR test row indices are missing or duplicated')


def sources():
    return dict(source_hashes(), **{'tools/v5_final_reading.py': sha256(__file__)})


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def external_artifacts(config):
    from huggingface_hub import snapshot_download
    cfg = config['recognizer']
    if not cfg.get('revision'):
        raise ValueError('Final evaluation requires an explicit public recognizer revision')
    snapshot = Path(snapshot_download(repo_id=cfg['name'], revision=cfg['revision'], local_files_only=True))
    files = {str(path): sha256(path) for path in sorted(snapshot.iterdir()) if path.is_file()}
    enhancer = build_enhancer(config.get('enhancer', {'type': 'identity'}))
    details = getattr(enhancer, 'provenance', {})
    if details.get('type') == 'self_dace':
        if details.get('source_revision') != SELF_DACE_REVISION:
            raise ValueError('Self-DACE checkout is not the pinned public revision')
        for key in ('source_file', 'checkpoint', 'denoise_checkpoint'):
            if details.get(key):
                files[details[key]] = sha256(details[key])
    if config.get('objective', {}).get('type') == 'reading_joint':
        from seeread.v5.reading_objective import LOSS_SOURCE, LOSS_SHA256
        path = Path(config['enhancer']['source_root']) / LOSS_SOURCE
        if sha256(path) != LOSS_SHA256:
            raise ValueError('External Reading LLE source changed')
        files[str(path)] = LOSS_SHA256
    return files


def freeze(config_path, protocol_path, baseline, output, checkpoint=None):
    config = plain(load_config(config_path))
    if checkpoint is None:
        if config['recognizer'].get('trainability') != 'frozen' or not config.get('enhancer', {}).get('freeze', True):
            raise ValueError('A trained baseline requires its dev-selected V5 checkpoint')
    else:
        if Path(checkpoint).name != 'best.pt':
            raise ValueError('Final evaluation must lock the complete-dev-selected best.pt, not last.pt')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        validate_checkpoint(saved, config)
        if saved['state'].get('best_cer') is None:
            raise ValueError('Checkpoint has no eligible complete dev selection')
        if saved.get('source_hashes') != source_hashes() or saved.get('runtime') != runtime_versions():
            raise ValueError('Final checkpoint must come from the locked current V5 implementation/runtime')
        selected = [v for v in saved.get('report', {}).get('validations', [])
                    if v.get('complete') and v['global_step'] == saved['state']['global_step']]
        if not selected or selected[-1]['raw']['cer'] != saved['state']['best_cer']:
            raise ValueError('Checkpoint cannot be traced to a complete dev-selected best evaluation')
        ids = saved['provenance']['recognizer'].get('decoder_start_token_ids', {})
        if ids.get('training') is None or ids.get('training') != ids.get('generation'):
            raise ValueError('Legacy wrong-prefix pilots are not eligible for reproduction final evaluation')
        del saved
    manifest = Path('data/annotations/v5/reading/test.jsonl')
    data = json.loads(DATA_PROVENANCE.read_text())
    if data['revision'] != DATA_REVISION or data['manifests']['test']['sha256'] != TEST_MANIFEST_SHA256:
        raise ValueError('Verified official data preparation provenance differs')
    # Validate the exact test source, not any recognition predictions.
    LockedLSTRTest(manifest, TEST_MANIFEST_SHA256, root=config['data'].get('root', 'data'), verify_hashes=True)
    protocol = dict(format=FORMAT, baseline=baseline, config=config,
                    config_source=str(config_path), config_sha256=sha256(config_path),
                    test_manifest=str(manifest), test_manifest_sha256=sha256(manifest),
                    checkpoint=str(checkpoint) if checkpoint else None,
                    checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
                    output=str(output), source_hashes=sources(), runtime=runtime_versions(),
                    data_provenance_sha256=sha256(DATA_PROVENANCE), external_artifacts=external_artifacts(config),
                    selection='locked before test; subsequent method selection uses dev only',
                    real_test_policy='ESTR60 deferred until the final project method is locked')
    protocol['protocol_sha256'] = digest(protocol)
    with open(protocol_path, 'x') as stream:
        json.dump(protocol, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return protocol


def evaluate_locked(path):
    protocol = json.loads(Path(path).read_text())
    stored = protocol.pop('protocol_sha256')
    if protocol.get('format') != FORMAT or digest(protocol) != stored:
        raise ValueError('Locked baseline protocol was modified')
    if protocol['source_hashes'] != sources() or protocol['runtime'] != runtime_versions():
        raise ValueError('Source or runtime differs from the locked baseline')
    if sha256(DATA_PROVENANCE) != protocol['data_provenance_sha256']:
        raise ValueError('Verified dataset provenance changed after protocol lock')
    for name, expected in protocol['external_artifacts'].items():
        if sha256(name) != expected:
            raise ValueError(f'Locked external source or public weights changed: {name}')
    checkpoint = protocol['checkpoint']
    if checkpoint and sha256(checkpoint) != protocol['checkpoint_sha256']:
        raise ValueError('Checkpoint bytes changed after protocol lock')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0', '2') or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose only authorized GPU2 as logical cuda:0; CPU fallback is disabled')
    config = protocol['config']
    torch.cuda.set_per_process_memory_fraction(float(config['train'].get('memory_fraction', .35)), 0)
    seed = int(config['train'].get('seed', 0))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dataset = LockedLSTRTest(protocol['test_manifest'], protocol['test_manifest_sha256'],
                            root=config['data'].get('root', 'data'), input_hw=config['data']['input_hw'], verify_hashes=True)
    output = Path(protocol['output'])
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Preserving existing final evaluation artifacts; output must be empty')
    # Exclusive reservation: even a second process cannot duplicate a final run.
    reservation = output / 'final.reserved.json'
    with open(reservation, 'x') as stream:
        json.dump(dict(protocol=str(path), protocol_sha256=stored), stream)
    report = dict(format=FORMAT, baseline=protocol['baseline'], status='initializing',
                  protocol=str(path), protocol_sha256=stored, dataset=dataset.provenance,
                  metric_note='raw and lower_alnum; paper normalization is not fully specified',
                  source_hashes=protocol['source_hashes'], runtime=protocol['runtime'])
    write_json(output / 'run.json', report)
    started = time.monotonic()
    try:
        model = build_pipeline(config).to('cuda:0')
        if checkpoint:
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            validate_checkpoint(saved, config)
            model.load_state_dict(saved['model'], strict=True)
            del saved
        report['provenance'] = provenance(model)
        result = evaluate(model, dataset, 'cuda:0', int(config['train']['batch_size']), output / 'test.records.jsonl')
        report.update(status='complete', result=result, elapsed_seconds=time.monotonic() - started)
        write_json(output / 'run.json', report)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic() - started)
        write_json(output / 'run.json', report)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('freeze')
    p.add_argument('--config', required=True)
    p.add_argument('--protocol', required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--checkpoint')
    p = sub.add_parser('evaluate')
    p.add_argument('--protocol', required=True)
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze(args.config, args.protocol, args.baseline, args.output, args.checkpoint)
    else:
        evaluate_locked(args.protocol)


if __name__ == '__main__':
    main()
