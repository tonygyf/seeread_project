"""Independent public-pretrained word reading baseline; dev raw CER selects weights.

Run inside the project container with GPU 2 exposed as logical cuda:0.
Raw and lower_alnum metrics reuse the same saved per-word predictions.
"""
from __future__ import annotations
import argparse
import fcntl
from importlib import metadata
import json
import os
from pathlib import Path
import random
import sys
import time

# The launcher owns physical GPU visibility; a GPU-2-only container normally
# exposes it as logical CUDA_VISIBLE_DEVICES=0. Never rewrite that mapping.
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from seeread.config import load_config
from seeread.v5.reading import WordCropDataset, check_disjoint, collate_words, metrics, sha256
from seeread.v5.recognition import build_pipeline, parameter_audit
from seeread.v5.training_safety import record_amp_rejection, reject_nonfinite_norm

FORMAT = 'seeread_v5_reading_v1'


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, '__dict__'):
        return plain(vars(value))
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA RNG device count differs from checkpoint')
        torch.cuda.set_rng_state_all(state['cuda'])


def source_hashes():
    paths = [Path(__file__).resolve(), ROOT / 'seeread/config.py',
             ROOT / 'seeread/models/recognizer.py', ROOT / 'seeread/metrics/spotting.py']
    paths.extend(ROOT / 'seeread/v5' / name for name in
                 ('__init__.py', 'enhancers.py', 'reading.py', 'recognition.py', 'reading_objective.py', 'training_safety.py'))
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def runtime_versions():
    versions = dict(torch=torch.__version__, cuda=torch.version.cuda,
                    cuda_allocator_config=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    cudnn_deterministic=torch.backends.cudnn.deterministic,
                    cudnn_benchmark=torch.backends.cudnn.benchmark)
    for name in ('transformers', 'peft', 'numpy', 'opencv-python'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def validate_resume(saved, config, manifests, sources, runtime, device):
    validate_checkpoint(saved, config)
    if saved.get('resume_safe') is not True:
        raise ValueError('Unsafe or unmarked diagnostic checkpoint cannot resume training')
    required = {'optimizer', 'scaler', 'rng', 'loader_rng', 'state', 'source_hashes', 'runtime', 'device', 'report'}
    if required - saved.keys():
        raise ValueError('Checkpoint lacks auditable exact-resume state/source hashes; start a fresh public-pretrained run')
    def recipe(value):
        value = plain(value)
        for field in ('max_seconds', 'max_steps', 'out_dir'):
            value.get('train', {}).pop(field, None)
        return value
    if recipe(saved['config']) != recipe(config):
        raise ValueError('Exact resume requires the same model/data/optimizer/training configuration')
    if saved['manifests'] != manifests:
        raise ValueError('Manifest contents or input protocol changed since checkpoint')
    if saved['source_hashes'] != sources:
        raise ValueError('Source SHA256 changed; exact resume refuses a different implementation')
    if saved['runtime'] != runtime or saved['device'] != device:
        raise ValueError('Runtime/library versions, deterministic settings, or logical device changed')


def provenance(model):
    return dict(recognizer=getattr(model.recognizer, 'provenance', {}),
                enhancer=getattr(model.enhancer, 'provenance', {'type': type(model.enhancer).__name__}),
                parameters=parameter_audit(model))


@torch.no_grad()
def evaluate(model, dataset, device, batch_size, records_path, deadline=None):
    model.eval()
    records = []
    with open(records_path, 'x', encoding='utf-8') as stream:
        for start in range(0, len(dataset), batch_size):
            if deadline is not None and time.monotonic() >= deadline:
                break
            images, rows = collate_words([dataset[i] for i in range(start, min(start + batch_size, len(dataset)))])
            predictions = model.predict(images.to(device))
            if len(predictions) != len(rows):
                raise ValueError('Recognizer returned the wrong number of predictions')
            for row, prediction in zip(rows, predictions):
                record = dict(id=row['id'], image=row['image'], text=row['text'], prediction=prediction,
                              source=row['source'], source_group=row['source_group'], split=row['split'])
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
    return dict(complete=len(records) == len(dataset), samples_requested=len(dataset),
                raw=metrics(records, 'raw'), lower_alnum=metrics(records, 'lower_alnum'),
                records=str(Path(records_path).resolve()))


def validate_checkpoint(saved, config):
    if saved.get('format') != FORMAT:
        raise ValueError('Only V5 public-initialized reading checkpoints are accepted; legacy SeeRead weights are forbidden')
    def topology(cfg):
        rec = dict(cfg['recognizer'])
        rec.pop('trainability', None)
        return rec, cfg.get('enhancer', {'type': 'identity'})
    if topology(saved['config']) != topology(config):
        raise ValueError('Checkpoint/config recognizer topology or enhancer differs')


def run(config, mode, device, checkpoint=None, pipeline_factory=build_pipeline, resume=None):
    """Own one output directory. Resume is explicit and limited to its last.pt."""
    config = plain(config)
    out = Path(config.get('train', {}).get('out_dir', 'results/v5/reading'))
    if resume and (mode != 'train' or checkpoint):
        raise ValueError('--resume is training-only and cannot be combined with --checkpoint')
    if resume and (Path(resume).name != 'last.pt' or Path(resume).resolve().parent != out.resolve()):
        raise ValueError('--resume must explicitly name this output directory\'s last.pt')
    out.mkdir(parents=True, exist_ok=True)
    with open(out / '.run.lock', 'a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another process already owns this reading run') from error
        if not resume and any(p.name != '.run.lock' for p in out.iterdir()):
            raise FileExistsError(f'Output directory is not empty: {out}')
        return _run(config, mode, device, checkpoint, pipeline_factory, resume)


def _run(config, mode, device, checkpoint, pipeline_factory, resume):
    config = plain(config)
    train_cfg, data_cfg = config.get('train', {}), config['data']
    eval_cfg = config.get('evaluation', {})
    if eval_cfg.get('selection_metric', 'raw_cer') != 'raw_cer':
        raise ValueError('V5 reading preregisters raw_cer for dev selection')
    # Validate both policies before building/downloading any model.
    metrics([], eval_cfg.get('normalization', 'raw'))
    if device.startswith('cuda'):
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if ',' in visible or not visible:
            raise ValueError('Expose exactly one GPU (the authorized GPU 2) using CUDA_VISIBLE_DEVICES')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; CPU fallback is disabled')
        if device not in ('cuda', 'cuda:0') or torch.cuda.device_count() != 1:
            raise ValueError('Use logical cuda:0 with exactly one visible GPU')
        fraction = float(train_cfg.get('memory_fraction', .7))
        if not 0 < fraction <= 1:
            raise ValueError('memory_fraction must be in (0,1]')
        torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    if mode not in ('train', 'evaluate'):
        raise ValueError(f'Unknown reading mode: {mode!r}')
    if mode == 'train' and checkpoint:
        raise ValueError('Training starts from public pretrained weights; --checkpoint is evaluation-only')
    seed = int(train_cfg.get('seed', 0))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    out = Path(train_cfg.get('out_dir', 'results/v5/reading'))
    input_hw = data_cfg.get('input_hw', [384, 384])
    kwargs = dict(root=data_cfg.get('root', 'data'), input_hw=input_hw,
                  verify_hashes=bool(data_cfg.get('verify_hashes', True)))
    dev = WordCropDataset(data_cfg['dev_manifest'], split='dev', **kwargs)
    train = WordCropDataset(data_cfg['train_manifest'], split='train', **kwargs) if mode == 'train' else None
    if train is not None:
        check_disjoint(train, dev)
    manifest_provenance = dict(dev=dev.provenance, train=train.provenance if train is not None else None)
    sources, runtime = source_hashes(), runtime_versions()
    resumed = torch.load(resume, map_location='cpu', weights_only=False) if resume else None
    if resumed is not None:
        validate_resume(resumed, config, manifest_provenance, sources, runtime, device)
    model = pipeline_factory(config).to(device)
    if resumed is not None and resumed['provenance'] != provenance(model):
        raise ValueError('Public model/enhancer source artifacts or parameter topology changed')
    loaded_provenance = None
    if checkpoint:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        validate_checkpoint(saved, config)
        model.load_state_dict(saved['model'], strict=True)
        loaded_provenance = saved['provenance']
        del saved
    report = dict(format=FORMAT, config=config, mode=mode, device=device,
                  runtime=runtime, source_hashes=sources,
                  visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), seed=seed,
                  provenance=provenance(model), loaded_checkpoint_provenance=loaded_provenance,
                  checkpoint=str(Path(checkpoint).resolve()) if checkpoint else None,
                  manifests=manifest_provenance, selection_metric='dev raw CER; paper normalization unspecified',
                  normalization_protocol='raw preserves case, punctuation and whitespace; lower_alnum is secondary',
                  status='running', validations=[], invocations=[])
    if resumed is not None:
        model.load_state_dict(resumed['model'], strict=True)
        report = plain(resumed['report'])
        report.update(config=config, status='running', resume=str(Path(resume).resolve()),
                      runtime=runtime, source_hashes=sources)
        for obsolete in ('error', 'stop_reason', 'state'):
            report.pop(obsolete, None)
    started = time.monotonic()
    batch_size = int(train_cfg.get('batch_size', 8))
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if mode == 'evaluate':
        write_json(out / 'run.json', report)
        try:
            result = evaluate(model, dev, device, batch_size, out / 'dev.records.jsonl')
            report.update(status='complete', result=result, elapsed_seconds=time.monotonic() - started)
            write_json(out / 'run.json', report)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return report
        except BaseException as error:
            report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                          error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic() - started)
            write_json(out / 'run.json', report)
            raise
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError('No trainable parameters; choose LoRA/full recognizer or a trainable enhancer')
    max_steps = int(train_cfg.get('max_steps', 1000))
    max_seconds = float(train_cfg.get('max_seconds', 1200))
    if max_steps < 0 or max_seconds < 0 or not (max_steps or max_seconds):
        raise ValueError('Training needs a positive max_steps or max_seconds budget')
    deadline = started + max_seconds if max_seconds else None
    eval_every = int(train_cfg.get('eval_every', 100))
    if eval_every <= 0:
        raise ValueError('eval_every must be positive')
    optimizer = torch.optim.AdamW(parameters, lr=float(train_cfg.get('lr', 1e-4)),
                                  weight_decay=float(train_cfg.get('weight_decay', 1e-4)))
    amp = bool(train_cfg.get('amp', True)) and device.startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    loader_rng = torch.Generator().manual_seed(seed)
    state = dict(global_step=0, epoch=0, cursor=0, order=None, amp_skips=0, amp_skip_streak=0, best_cer=None)
    if resumed is not None:
        optimizer.load_state_dict(resumed['optimizer'])
        scaler.load_state_dict(resumed['scaler'])
        loader_rng.set_state(resumed['loader_rng'])
        state = plain(resumed['state'])
        # Model/optimizer construction may consume RNG. Restore only after all
        # stateful objects have been constructed and their weights loaded.
        restore_rng(resumed['rng'])
        del resumed
    invocation = dict(index=len(report['invocations']), resumed=bool(resume),
                      start_step=state['global_step'], max_steps_total=max_steps,
                      max_seconds_this_invocation=max_seconds, status='running')
    report['invocations'].append(invocation)
    write_json(out / 'run.json', report)
    events = open(out / 'train_events.jsonl', 'a' if resume else 'x')

    def event(kind, **values):
        events.write(json.dumps(dict(event=kind, global_step=state['global_step'],
                                     invocation=invocation['index'],
                                     elapsed_seconds=time.monotonic() - started, **values), allow_nan=False) + '\n')
        events.flush()

    def save(name):
        payload = dict(format=FORMAT, resume_safe=name != 'interrupted.pt', config=config,
                       model=model.state_dict(), optimizer=optimizer.state_dict(),
                       scaler=scaler.state_dict(), rng=rng_state(), loader_rng=loader_rng.get_state(),
                       state=state.copy(), provenance=report['provenance'], manifests=manifest_provenance,
                       selection_metric='dev raw CER', scheduler=None, source_hashes=sources,
                       runtime=runtime, device=device, report=report)
        temporary = out / (name + '.tmp')
        torch.save(payload, temporary)
        temporary.replace(out / name)

    def validate():
        record_path = out / f'dev_step{state["global_step"]:07d}.records.jsonl'
        attempt = 0
        while record_path.exists():
            attempt += 1
            record_path = out / f'dev_step{state["global_step"]:07d}.attempt{attempt:03d}.records.jsonl'
        result = evaluate(model, dev, device, batch_size,
                          record_path, deadline)
        report['validations'].append(dict(global_step=state['global_step'], **result))
        cer = result['raw']['cer']
        eligible = result['complete'] and cer is not None
        event('validation', eligible_for_selection=eligible, **result)
        if eligible and (state['best_cer'] is None or cer < state['best_cer']):
            state['best_cer'] = cer
            save('best.pt')
        write_json(out / 'run.json', report)

    try:
        event('resume' if resume else 'start', budgets=invocation.copy())
        save('last.pt')
        if not resume and bool(train_cfg.get('eval_initial', True)):
            validate()
        while (not max_steps or state['global_step'] < max_steps) and (deadline is None or time.monotonic() < deadline):
            if state['order'] is None or state['cursor'] >= len(train):
                if state['order'] is not None:
                    state['epoch'] += 1
                state['order'] = torch.randperm(len(train), generator=loader_rng).tolist()
                state['cursor'] = 0
            end = min(state['cursor'] + batch_size, len(train))
            indices = state['order'][state['cursor']:end]
            images, rows = collate_words([train[i] for i in indices])
            model.train()
            buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda' if device.startswith('cuda') else 'cpu', enabled=amp):
                loss = model.forward_loss(images.to(device), [r['text'] for r in rows])
            if loss is None or not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite/missing reading loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 5.)
            if reject_nonfinite_norm(norm, amp=amp, scaler=scaler, optimizer=optimizer, state=state,
                                     event=event, max_skips=int(train_cfg.get('max_amp_skips', 20))):
                with torch.no_grad():
                    for name, value in model.named_buffers():
                        value.copy_(buffers[name])
                continue
            before = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            if amp and scaler.get_scale() < before:
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for name, value in model.named_buffers():
                        value.copy_(buffers[name])
                record_amp_rejection(state, int(train_cfg.get('max_amp_skips', 20)), event,
                                     scale=scaler.get_scale(), reason='scaler_rejected_update')
                continue
            state['amp_skip_streak'] = 0
            state['cursor'] = end
            state['global_step'] += 1
            event('update', loss=float(loss.detach()), gradient_norm=float(norm), samples=len(rows),
                  components=getattr(model, 'last_loss_components', None))
            if state['global_step'] % eval_every == 0:
                save('last.pt')
                validate()
                save('last.pt')
        # Always attempt a final validation; an exhausted wall-clock budget
        # produces a partial report which is never eligible for best selection.
        save('last.pt')
        if (not report['validations'] or report['validations'][-1]['global_step'] != state['global_step']
                or not report['validations'][-1]['complete']):
            validate()
        invocation.update(status='complete', end_step=state['global_step'], elapsed_seconds=time.monotonic() - started)
        report.update(status='complete', stop_reason='step_budget' if max_steps and state['global_step'] >= max_steps else 'time_budget',
                      state=state, elapsed_seconds=time.monotonic() - started)
        save('last.pt')
        write_json(out / 'run.json', report)
        return report
    except BaseException as error:
        invocation.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                          end_step=state['global_step'], elapsed_seconds=time.monotonic() - started)
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error=f'{type(error).__name__}: {error}', state=state, elapsed_seconds=time.monotonic() - started)
        save('interrupted.pt')
        write_json(out / 'run.json', report)
        raise
    finally:
        events.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=('evaluate', 'train'), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--checkpoint', help='V5 reading checkpoint, evaluation only')
    parser.add_argument('--resume', help='Exact V5 training resume from this run\'s last.pt')
    parser.add_argument('--max-seconds', type=float, help='New wall-clock budget for this invocation')
    parser.add_argument('--max-steps', type=int, help='Total optimizer-update cap, including resumed steps')
    args = parser.parse_args()
    cfg = plain(load_config(args.config))
    for name in ('max_seconds', 'max_steps'):
        if getattr(args, name) is not None:
            cfg.setdefault('train', {})[name] = getattr(args, name)
    run(cfg, args.mode, args.device, args.checkpoint, resume=args.resume)


if __name__ == '__main__':
    main()
