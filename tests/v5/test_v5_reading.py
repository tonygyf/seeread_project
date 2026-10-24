"""V5 reading contract tests: no pretrained downloads and no CUDA execution."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from seeread.v5.reading import WordCropDataset, check_disjoint, metrics, sha256
from seeread.v5.recognition import ReadingTrOCR, build_recognizer, parameter_audit, build_pipeline
from seeread.v5.training_safety import reject_nonfinite_norm
from tools.v5_reading import run, validate_checkpoint


class DummyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.dropout = nn.Dropout(.8)
        self.config = SimpleNamespace(_commit_hash='public-commit', decoder_start_token_id=0, pad_token_id=1,
                                      to_dict=lambda: {'architecture': 'dummy'})
        self.generation_config = SimpleNamespace(decoder_start_token_id=2, bos_token_id=0,
                                                  to_dict=lambda: {'num_beams': 1, 'decoder_start_token_id': 2})


class DummyDonor:
    def __init__(self, **kwargs):
        self.base, self.tok, self.max_len = DummyBase(), None, kwargs['max_len']


def test_training_prefix_matches_public_generation_and_frozen_decode_is_preserved():
    from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import shift_tokens_right
    with patch('seeread.v5.recognition.TrOCRRecognizer', DummyDonor):
        for trainability in ('frozen', 'full'):
            model = ReadingTrOCR({'adapter': 'none', 'trainability': trainability})
            assert model.base.generation_config.decoder_start_token_id == 2
            assert model.base.config.decoder_start_token_id == 2
            labels = torch.tensor([[0, 42, 2]])
            decoder_inputs = shift_tokens_right(labels, model.base.config.pad_token_id,
                                                model.base.config.decoder_start_token_id)
            assert decoder_inputs.tolist() == [[2, 0, 42]]
            assert model.provenance['decoder_start_token_ids']['training'] == 2
            assert model.provenance['decoder_start_token_ids']['generation'] == 2


def test_topology_and_freezing_are_independent_and_parameters_are_unique():
    def inject(base, cfg):
        base.register_parameter('lora_A', nn.Parameter(torch.ones(2)))
        return nn.Module()  # Wrapper must not become another registered copy.
    peft = SimpleNamespace(LoraConfig=lambda **kw: SimpleNamespace(**kw), get_peft_model=inject)
    with patch('seeread.v5.recognition.TrOCRRecognizer', DummyDonor), patch.dict(sys.modules, {'peft': peft}):
        frozen = ReadingTrOCR({'adapter': 'lora', 'trainability': 'frozen'})
        trained = ReadingTrOCR({'adapter': 'lora', 'trainability': 'lora'})
        assert set(frozen.state_dict()) == set(trained.state_dict())
        frozen.load_state_dict(trained.state_dict(), strict=True)
        assert 'base.lora_A' in frozen.state_dict()
        frozen.train()
        assert not frozen.training and not frozen.base.dropout.training
        assert parameter_audit(frozen)['trainable_parameters'] == 0
        assert parameter_audit(trained)['trainable_names'] == ['base.lora_A']
        assert len(list(trained.parameters())) == len({id(p) for p in trained.parameters()})
        assert not any('peft.' in n for n in trained.state_dict())


def test_invalid_recognizer_enums_fail_before_pretrained_loading():
    with patch('seeread.v5.recognition.TrOCRRecognizer', side_effect=AssertionError('network load attempted')):
        for cfg in ({'type': 'trcor'}, {'type': 'trocr', 'adapter': 'lorra'},
                    {'type': 'trocr', 'trainability': 'train'}, {'type': 'trocr', 'adapter': 'none', 'trainability': 'lora'}):
            try:
                build_recognizer(cfg)
            except ValueError:
                pass
            else:
                raise AssertionError(f'Invalid config accepted: {cfg}')


def test_metrics_preserve_raw_characters_and_save_secondary_policy():
    records = [{'text': 'A !', 'prediction': 'a!'}, {'text': 'B', 'prediction': 'B'}]
    raw, secondary = metrics(records), metrics(records, 'lower_alnum')
    assert raw['samples'] == 2 and raw['reference_characters'] == 4
    assert raw['edit_operations'] == 2 and raw['cer'] == .5 and raw['word_accuracy'] == .5
    assert secondary['cer'] == 0 and secondary['word_accuracy'] == 1
    try:
        metrics([], 'typo')
    except ValueError:
        pass
    else:
        raise AssertionError('Unknown normalization accepted for empty results')


def prepare(root, split, value=120):
    image = root / f'{split}.png'
    cv2.imwrite(str(image), np.full((12, 32, 3), value, np.uint8))
    row = dict(id=split + ':1', image=image.name, lit=None, text='A!', source='public-test-source',
               source_group=split + '-group', split=split, dataset={'repo': 'public/repo', 'revision': 'abc'},
               sha256={'image': sha256(image), 'lit': None}, scene_group_verified=False)
    manifest = root / f'{split}.jsonl'
    manifest.write_text(json.dumps(row) + '\n')
    return manifest, row


def test_data_rejects_test_splits_and_content_leakage():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train_path, train_row = prepare(root, 'train', 80)
        dev_path, dev_row = prepare(root, 'dev', 180)
        train = WordCropDataset(train_path, root, 'train')
        dev = WordCropDataset(dev_path, root, 'dev')
        check_disjoint(train, dev)
        test_path, _ = prepare(root, 'test')
        real_path, _ = prepare(root, 'test_real')
        for path in (test_path, real_path):
            for expected in ('train', 'dev'):
                try:
                    WordCropDataset(path, root, expected)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Test data accepted for training/model selection')
        dev.rows[0]['sha256']['image'] = train_row['sha256']['image']
        try:
            check_disjoint(train, dev)
        except ValueError:
            pass
        else:
            raise AssertionError('Duplicate image content accepted across train/dev')


class TinyReading(nn.Module):
    def __init__(self):
        super().__init__()
        self.enhancer = nn.Identity()
        self.recognizer = nn.Linear(1, 1, bias=False)
        self.recognizer.provenance = {'public_pretrained': 'unit-test', 'actual_commit_hash': 'test-public'}
        self.prediction_calls = 0

    def forward_loss(self, crops, texts):
        return (self.recognizer.weight - crops.mean()).square().sum()

    def predict(self, crops):
        self.prediction_calls += 1
        return ['a!'] * len(crops)


def config(root, out):
    train, _ = prepare(root, 'train', 80)
    dev, _ = prepare(root, 'dev', 180)
    return dict(recognizer={'type': 'trocr', 'adapter': 'lora', 'trainability': 'lora'}, enhancer={'type': 'identity'},
                data={'root': str(root), 'train_manifest': str(train), 'dev_manifest': str(dev), 'input_hw': [16, 32]},
                train={'out_dir': str(out), 'max_steps': 2, 'max_seconds': 30, 'batch_size': 1,
                       'eval_every': 1, 'amp': False, 'lr': .01})


def test_r0_evaluation_uses_one_prediction_for_both_metrics_and_raw_records():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = config(root, root / 'eval')
        del cfg['data']['train_manifest']  # R0 must work with dev alone.
        model = TinyReading()
        report = run(cfg, 'evaluate', 'cpu', pipeline_factory=lambda _: model)
        assert model.prediction_calls == 1
        assert report['result']['raw']['cer'] == .5
        assert report['result']['lower_alnum']['cer'] == 0
        records = [json.loads(line) for line in (root / 'eval/dev.records.jsonl').read_text().splitlines()]
        assert records[0]['text'] == 'A!' and records[0]['prediction'] == 'a!'
        assert report['manifests']['train'] is None


def test_short_training_saves_complete_v5_state_and_never_loads_legacy_weights():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = config(root, root / 'train_out')
        model = TinyReading()
        before = model.recognizer.weight.detach().clone()
        report = run(cfg, 'train', 'cpu', pipeline_factory=lambda _: model)
        assert report['state']['global_step'] == 2
        assert not torch.equal(before, model.recognizer.weight)
        saved = torch.load(root / 'train_out/last.pt', weights_only=False)
        assert {'config', 'optimizer', 'rng', 'loader_rng', 'provenance', 'manifests', 'state', 'model'} <= saved.keys()
        assert saved['optimizer']['state']
        assert saved['state']['global_step'] == 2
        assert saved['selection_metric'] == 'dev raw CER'
        assert (root / 'train_out/best.pt').exists()
        try:
            validate_checkpoint({'det': {}, 'rec': {}, 'enh': {}}, cfg)
        except ValueError:
            pass
        else:
            raise AssertionError('Legacy checkpoint accepted')


class StochasticReading(TinyReading):
    def forward_loss(self, crops, texts):
        import random
        noise = .01 * (torch.rand(()) + np.random.random() + random.random())
        return (self.recognizer.weight - crops.mean() - noise).square().sum()


def _four_training_words(root, cfg):
    rows = []
    for index, value in enumerate((30, 70, 110, 150)):
        image = root / f'word{index}.png'
        cv2.imwrite(str(image), np.full((12, 32, 3), value, np.uint8))
        rows.append(dict(id=f'train:{index}', image=image.name, lit=None, text='A!', source='public-test-source',
                         source_group=f'train-group-{index}', split='train',
                         dataset={'repo': 'public/repo', 'revision': 'abc'},
                         sha256={'image': sha256(image), 'lit': None}))
    Path(cfg['data']['train_manifest']).write_text(''.join(json.dumps(r) + '\n' for r in rows))


def test_resume_mid_epoch_preserves_rng_optimizer_and_loader_trajectory():
    import copy
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = config(root, root / 'full')
        _four_training_words(root, cfg)
        cfg['train'].update(max_steps=4, eval_every=2, seed=17)
        run(cfg, 'train', 'cpu', pipeline_factory=lambda _: StochasticReading())
        full = torch.load(root / 'full/last.pt', weights_only=False)
        split_cfg = copy.deepcopy(cfg)
        split_cfg['train'].update(out_dir=str(root / 'split'), max_steps=2)
        run(split_cfg, 'train', 'cpu', pipeline_factory=lambda _: StochasticReading())
        mid = torch.load(root / 'split/last.pt', weights_only=False)
        assert mid['state']['cursor'] == 2 and mid['state']['epoch'] == 0
        old_records = {p.name: p.read_bytes() for p in (root / 'split').glob('*.records.jsonl')}
        old_events = (root / 'split/train_events.jsonl').read_bytes()
        split_cfg['train'].update(max_steps=4, max_seconds=60)
        resumed = run(split_cfg, 'train', 'cpu', resume=root / 'split/last.pt',
                      pipeline_factory=lambda _: StochasticReading())
        saved = torch.load(root / 'split/last.pt', weights_only=False)
        assert saved['state'] == full['state']
        assert torch.equal(saved['rng']['torch'], full['rng']['torch'])
        assert saved['rng']['python'] == full['rng']['python']
        assert np.array_equal(saved['rng']['numpy'][1], full['rng']['numpy'][1])
        assert torch.equal(saved['loader_rng'], full['loader_rng'])
        for key in full['model']:
            assert torch.equal(saved['model'][key], full['model'][key]), key
        for key, value in full['optimizer']['state'][0].items():
            assert torch.equal(saved['optimizer']['state'][0][key], value), key
        assert (root / 'split/train_events.jsonl').read_bytes().startswith(old_events)
        assert all((root / 'split' / name).read_bytes() == data for name, data in old_records.items())
        assert [v['global_step'] for v in resumed['validations']] == [0, 2, 4]
        assert len(resumed['invocations']) == 2
        assert resumed['invocations'][1]['max_seconds_this_invocation'] == 60


def test_resume_rejects_changed_code_recipe_or_manifest_without_overwriting_run():
    import copy
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = config(root, root / 'run')
        run(cfg, 'train', 'cpu', pipeline_factory=lambda _: TinyReading())
        before = (root / 'run/run.json').read_bytes()
        changed = copy.deepcopy(cfg)
        changed['train']['lr'] = .2
        for override, changed_sources in ((changed, False), (cfg, True)):
            patches = patch('tools.v5_reading.source_hashes', return_value={'fake.py': 'changed'}) if changed_sources else nullcontext()
            with patches:
                try:
                    run(override, 'train', 'cpu', resume=root / 'run/last.pt',
                        pipeline_factory=lambda _: TinyReading())
                except ValueError:
                    pass
                else:
                    raise AssertionError('Changed recipe/source accepted for exact resume')
        assert (root / 'run/run.json').read_bytes() == before
        try:
            run(cfg, 'train', 'cpu', resume=root / 'run/best.pt', pipeline_factory=lambda _: TinyReading())
        except ValueError:
            pass
        else:
            raise AssertionError('Exact resume accepted a checkpoint other than this run last.pt')
        manifest = Path(cfg['data']['dev_manifest'])
        row = json.loads(manifest.read_text())
        row['text'] = 'CHANGED'
        manifest.write_text(json.dumps(row) + '\n')
        try:
            run(cfg, 'train', 'cpu', resume=root / 'run/last.pt', pipeline_factory=lambda _: TinyReading())
        except ValueError:
            pass
        else:
            raise AssertionError('Changed manifest accepted for exact resume')
        assert (root / 'run/run.json').read_bytes() == before


def test_resume_retries_partial_validation_without_overwriting_records():
    from tools import v5_reading as cli
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = config(root, root / 'run')
        cfg['train'].update(max_steps=1, eval_every=100, eval_initial=False)
        original = cli.evaluate
        def partial(*args, **kwargs):
            report = original(*args, **kwargs)
            report['complete'] = False
            return report
        with patch.object(cli, 'evaluate', partial):
            run(cfg, 'train', 'cpu', pipeline_factory=lambda _: TinyReading())
        record = root / 'run/dev_step0000001.records.jsonl'
        old = record.read_bytes()
        result = run(cfg, 'train', 'cpu', resume=root / 'run/last.pt', pipeline_factory=lambda _: TinyReading())
        assert result['state']['global_step'] == 1
        assert result['validations'][-1]['complete']
        assert record.read_bytes() == old
        assert (root / 'run/dev_step0000001.attempt001.records.jsonl').exists()
        assert (root / 'run/best.pt').exists()


class InitializationTests(unittest.TestCase):
    def test_enhancer_construction_cannot_change_recognizer_initialization(self):
        def enhancer(cfg):
            if cfg['type'] == 'identity':
                return nn.Identity()
            return nn.Sequential(nn.Conv2d(3, 32, 3), nn.Conv2d(32, 3, 3))
        def recognizer(cfg):
            return nn.Linear(32, 10)
        with patch('seeread.v5.enhancers.build_enhancer', enhancer), \
             patch('seeread.v5.recognition.build_recognizer', recognizer):
            torch.manual_seed(0)
            a = build_pipeline(dict(enhancer={'type': 'identity'}, recognizer={}))
            after_a = torch.get_rng_state()
            torch.manual_seed(0)
            b = build_pipeline(dict(enhancer={'type': 'learned'}, recognizer={}))
            after_b = torch.get_rng_state()
        for name, tensor in a.recognizer.state_dict().items():
            self.assertTrue(torch.equal(tensor, b.recognizer.state_dict()[name]))
        self.assertTrue(torch.equal(after_a, after_b))

class NormSafetyTests(unittest.TestCase):
    def test_finite_gradients_with_overflowing_norm_never_update_adam(self):
        parameter = torch.nn.Parameter(torch.ones(2))
        optimizer = torch.optim.AdamW([parameter], lr=.1, weight_decay=.1)
        scaler = torch.amp.GradScaler('cpu')
        scaler.scale((parameter * 1e20).sum()).backward()
        scaler.unscale_(optimizer)
        self.assertTrue(torch.isfinite(parameter.grad).all())
        norm = torch.nn.utils.clip_grad_norm_([parameter], 5.)
        self.assertFalse(torch.isfinite(norm))
        old = parameter.detach().clone()
        old_scale = scaler.get_scale()
        state, events = dict(cursor=0, amp_skips=0), []
        skipped = reject_nonfinite_norm(norm, amp=True, scaler=scaler, optimizer=optimizer, state=state,
                                        event=lambda kind, **v: events.append((kind, v)), max_skips=3)
        self.assertTrue(skipped)
        self.assertEqual(scaler.get_scale(), old_scale / 2)
        self.assertTrue(torch.equal(old, parameter))
        self.assertEqual(len(optimizer.state), 0)
        self.assertIsNone(parameter.grad)
        self.assertEqual(state, dict(cursor=0, amp_skips=1, amp_skip_streak=1))
        self.assertEqual(events[0][1]['reason'], 'nonfinite_gradient_norm')

    def test_optimizer_failure_preserves_last_healthy_checkpoint(self):
        from tools import v5_reading as cli
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = config(root, root / 'run')
            cfg['train'].update(eval_initial=False)
            original = torch.optim.AdamW.step
            healthy = []

            def mutate_then_fail(optimizer, *args, **kwargs):
                healthy.append((root / 'run/last.pt').read_bytes())
                original(optimizer, *args, **kwargs)
                raise RuntimeError('injected failure after optimizer mutation')

            with patch.object(torch.optim.AdamW, 'step', mutate_then_fail):
                with self.assertRaisesRegex(RuntimeError, 'after optimizer mutation'):
                    run(cfg, 'train', 'cpu', pipeline_factory=lambda _: TinyReading())
            self.assertEqual((root / 'run/last.pt').read_bytes(), healthy[0])
            saved = torch.load(root / 'run/last.pt', weights_only=False)
            self.assertTrue(saved['resume_safe'])
            self.assertEqual(saved['state']['global_step'], 0)
            interrupted = torch.load(root / 'run/interrupted.pt', weights_only=False)
            self.assertFalse(interrupted['resume_safe'])
            with self.assertRaisesRegex(ValueError, 'diagnostic|unsafe'):
                cli.validate_resume(interrupted, cfg, saved['manifests'], saved['source_hashes'],
                                    saved['runtime'], 'cpu')


if __name__ == '__main__':
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
            print('PASS', name)

    unittest.main()
