"""Usage: python tools/legacy/train.py --config configs/legacy/seeread_joint.yaml [--set train.epochs=1]
Stages: det (detector only) -> rec (enhancer+recognizer on GT geometry, detector frozen) -> joint (all)."""
import argparse, os, random, sys, json, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from seeread.config import load_config
from seeread.data.datasets import SeeReadDataset, collate
from seeread.metrics.spotting import SpottingEvaluator
from seeread.models.recognizer import build_recognizer
from seeread.models.seeread import SeeRead


def parse_set(items):
    ov = {}
    for s in items or []:
        k, v = s.split("=", 1); d = ov
        *path, last = k.split(".")
        for p in path: d = d.setdefault(p, {})
        d[last] = json.loads(v) if v[:1] in "0123456789-[{tf" else v
    return ov


def set_frozen_eval(model, stage):
    """Keep parameter-frozen modules in eval() so BN running stats stay frozen even after model.train()."""
    if stage == "rec":
        model.det.eval()
    elif stage == "det":
        model.enh.eval(); model.rec.eval()
    elif stage == "joint" and getattr(model.cfg.train, "freeze_rec", False):
        model.rec.eval()


def config_dict(value):
    if isinstance(value, SimpleNamespace):
        return {k: config_dict(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {k: config_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [config_dict(v) for v in value]
    return value


def rng_state():
    numpy_state = np.random.get_state()
    # Keep checkpoint contents compatible with existing weights_only=True evaluators.
    numpy_state = dict(kind=numpy_state[0], keys=torch.tensor(numpy_state[1].astype(np.int64)),
                       position=numpy_state[2], has_gauss=numpy_state[3], cached_gauss=numpy_state[4])
    return dict(python=random.getstate(), numpy=numpy_state,
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns["kind"], ns["keys"].cpu().numpy().astype(np.uint32),
                         ns["position"], ns["has_gauss"], ns["cached_gauss"]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def load_checkpoint(path, device="cpu"):
    # Training checkpoints include Python/NumPy RNG state; load only trusted local files.
    return torch.load(path, map_location=device, weights_only=False)


def save_checkpoint(path, model, cfg, opt, sched, scaler, state, provenance):
    payload = dict(format_version=2, det=model.det.state_dict(), enh=model.enh.state_dict(),
                   rec=model.rec.state_dict(), cfg=config_dict(cfg.train), config=config_dict(cfg),
                   optimizer=opt.state_dict(), scheduler=sched.state_dict(), scaler=scaler.state_dict(),
                   rng=rng_state(), state=state, provenance=provenance)
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def check_resume_config(saved, current):
    # LR horizon, stage, data geometry and worker count must match for exact continuation.
    def comparable(cfg):
        cfg = json.loads(json.dumps(cfg))
        for key in ("out_dir", "resume", "resume_mode", "max_steps", "max_seconds",
                    "init_det", "init_rec", "checkpoint_every_steps"):
            cfg["train"].pop(key, None)
        return cfg
    if comparable(saved) != comparable(current):
        raise ValueError("Exact resume requires the checkpoint's training/data/model configuration; "
                         "use --resume-mode weights-only for an intentional new recipe")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--set", nargs="*")
    ap.add_argument("--resume", help="trusted checkpoint path")
    ap.add_argument("--resume-mode", choices=("exact", "weights-only"))
    ap.add_argument("--max-steps", type=int, help="additional optimizer updates in this invocation; 0 disables")
    ap.add_argument("--max-seconds", type=float, help="wall-time budget including validation; 0 disables")
    ap.add_argument("--val-max-batches", type=int, help="0 validates all batches; partial scores never select best.pt")
    ap.add_argument("--gpu-memory-fraction", type=float, help="per-process CUDA allocator limit, in (0, 1]")
    a = ap.parse_args(); cfg = load_config(a.config, parse_set(a.set)); t = cfg.train
    for field in ("resume", "resume_mode", "max_steps", "max_seconds", "val_max_batches", "gpu_memory_fraction"):
        value = getattr(a, field)
        if value is not None:
            setattr(t, field, value)
    resume_path = getattr(t, "resume", None)
    resume_mode = getattr(t, "resume_mode", "exact")
    max_steps = int(getattr(t, "max_steps", 0) or 0)
    max_seconds = float(getattr(t, "max_seconds", 0) or 0)
    val_max_batches = int(getattr(t, "val_max_batches", 0) or 0)
    if min(max_steps, max_seconds, val_max_batches) < 0:
        raise ValueError("Budgets must be nonnegative")
    out = Path(t.out_dir)
    checkpoint = load_checkpoint(resume_path) if resume_path else None
    if checkpoint is not None and resume_mode == "exact":
        if checkpoint.get("format_version") != 2:
            raise ValueError("Legacy checkpoint has weights only; use --resume-mode weights-only and a NEW out_dir")
        check_resume_config(checkpoint["config"], config_dict(cfg))
    existing = out.exists() and any(out.iterdir())
    if existing and not (checkpoint is not None and resume_mode == "exact" and
                         Path(resume_path).resolve().parent == out.resolve()):
        raise ValueError(f"Refusing to overwrite nonempty output directory: {out}; choose a NEW train.out_dir")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    random.seed(t.seed); np.random.seed(t.seed); torch.manual_seed(t.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if getattr(t, "require_cuda", False) and dev != "cuda":
        raise RuntimeError("train.require_cuda=true but CUDA is unavailable; refusing CPU fallback")
    gpu_memory_fraction = float(getattr(t, "gpu_memory_fraction", 1.0))
    if not 0 < gpu_memory_fraction <= 1:
        raise ValueError("gpu_memory_fraction must be in (0, 1]")
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
    amp_name = str(getattr(t, "amp_dtype", "float16"))
    if amp_name not in ("float16", "bfloat16"):
        raise ValueError("train.amp_dtype must be float16 or bfloat16")
    amp_dtype = getattr(torch, amp_name)
    early_stop_patience = int(getattr(t, "early_stop_patience", 0) or 0)
    if early_stop_patience < 0:
        raise ValueError("train.early_stop_patience must be nonnegative")
    amp_overflow_patience = int(getattr(t, "amp_overflow_patience", 16))
    if amp_overflow_patience < 1:
        raise ValueError("train.amp_overflow_patience must be positive")
    model = SeeRead(build_recognizer(cfg.rec), cfg).to(dev)
    if checkpoint is not None:
        for name in ("det", "enh", "rec"):
            getattr(model, name).load_state_dict(checkpoint[name])
    else:
        if t.init_det: model.det.load_state_dict(load_checkpoint(t.init_det)["det"])
        if t.init_rec:
            sd = load_checkpoint(t.init_rec); model.enh.load_state_dict(sd["enh"]); model.rec.load_state_dict(sd["rec"])
            del sd
    if t.stage == "rec":
        for p in model.det.parameters(): p.requires_grad = False
    elif t.stage == "det":
        for m in (model.enh, model.rec):
            for p in m.parameters(): p.requires_grad = False
    if getattr(t, "freeze_rec", False) and t.stage == "joint":   # S3-v3: protect the S2 recognizer
        for p in model.rec.parameters(): p.requires_grad = False
    loader_gen = torch.Generator().manual_seed(t.seed)
    if getattr(cfg.data, "train_anns", None):               # multi-dataset mixture (S3-v3)
        from seeread.data.mixture import build_mixture
        ds, sampler, _mix_sizes = build_mixture(
            cfg.data.root, cfg.data.train_anns, cfg.data.mix_weights,
            num_samples=int(getattr(cfg.data, "mix_samples_per_epoch", 0) or 0) or None,
            train=True, size=cfg.data.size, synth_mode=cfg.data.synth_mode, seed=t.seed)
        print(f"[mix] dataset sizes={_mix_sizes} weights={cfg.data.mix_weights} draws/epoch={len(sampler)}", flush=True)
        sampler.generator = loader_gen
        dl = DataLoader(ds, t.batch_size, sampler=sampler, num_workers=cfg.data.workers,
                        collate_fn=collate, drop_last=True, generator=loader_gen)
    else:
        ds = SeeReadDataset(cfg.data.root, cfg.data.train_ann, train=True, size=cfg.data.size, synth_mode=cfg.data.synth_mode, seed=t.seed)
        dl = DataLoader(ds, t.batch_size, shuffle=True, num_workers=cfg.data.workers, collate_fn=collate, drop_last=True, generator=loader_gen)
    if not len(dl):
        raise ValueError("Training loader has no complete batches; reduce train.batch_size")
    val_every = int(getattr(t, "val_every", 0) or 0)        # 0 disables validation; else eval every N epochs and keep best.pt
    val_dl = None
    if val_every > 0 and getattr(cfg.data, "val_ann", None):
        val_ds = SeeReadDataset(cfg.data.root, cfg.data.val_ann, train=False, size=cfg.data.size,
                               synth_mode=cfg.data.synth_mode, seed=t.seed,
                               eval_condition=getattr(t, "val_condition", "original"))
        val_dl = DataLoader(val_ds, 4, shuffle=False, num_workers=cfg.data.workers, collate_fn=collate,
                            generator=torch.Generator().manual_seed(t.seed))
    params = [p for p in model.parameters() if p.requires_grad]
    lr_rec = getattr(t, "lr_rec", None)                     # optional lower LR for the recognizer in joint FT
    groups = [{"params": params, "lr": t.lr}]
    if lr_rec and t.stage == "joint" and not getattr(t, "freeze_rec", False):
        ids = {id(p) for p in model.rec.parameters()}
        groups[0]["params"] = rest = [p for p in params if id(p) not in ids]
        rec_p = [p for p in params if id(p) in ids]
        if rec_p:
            groups.append({"params": rec_p, "lr": float(lr_rec)})
        print(f"[opt] rec group lr={lr_rec} ({len(rec_p)} tensors), rest lr={t.lr} ({len(rest)} tensors)", flush=True)
    opt = torch.optim.AdamW(groups, weight_decay=t.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, t.epochs * len(dl))
    scaler = torch.amp.GradScaler(enabled=t.amp and dev == "cuda" and amp_dtype == torch.float16)
    state = dict(epoch=0, batch_in_epoch=0, global_step=0, best_f1=-1.0,
                 empty_skips=0, amp_skips=0, amp_consecutive_skips=0,
                 loss_batches=0, cic_batches=0, val_no_improve=0)
    provenance = dict(mode=resume_mode if resume_path else "fresh", source=str(Path(resume_path).resolve()) if resume_path else None,
                      optimizer_reset=bool(resume_path and resume_mode == "weights-only"),
                      parent=checkpoint.get("provenance") if checkpoint else None,
                      source_train_config=checkpoint.get("cfg") if checkpoint else None)
    resume_rng = None
    if checkpoint is not None and resume_mode == "exact":
        opt.load_state_dict(checkpoint["optimizer"])
        # Checkpoints are loaded on CPU; Adam moments must follow their parameter device.
        for group in opt.param_groups:
            for parameter in group["params"]:
                for key, value in opt.state[parameter].items():
                    if torch.is_tensor(value) and key != "step":
                        opt.state[parameter][key] = value.to(parameter.device)
        sched.load_state_dict(checkpoint["scheduler"]); scaler.load_state_dict(checkpoint["scaler"])
        state.update(checkpoint["state"])
        loader_gen.set_state(state["loader_rng"])
        resume_rng = checkpoint["rng"]
        restore_rng(resume_rng)
    # Do not retain an extra CPU copy of the source model/Adam moments during training.
    del checkpoint
    invocation_step = state["global_step"]
    total = t.epochs * len(dl)
    checkpoint_every = int(getattr(t, "checkpoint_every_steps", 100) or 0)

    def event(kind, **fields):
        record = dict(event=kind, epoch=state["epoch"], batch_in_epoch=state["batch_in_epoch"],
                      global_step=state["global_step"], elapsed_seconds=time.monotonic() - started, **fields)
        with (out / "train_events.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    with (out / "run.json").open("w") as stream:
        json.dump(dict(config=config_dict(cfg), provenance=provenance, budgets=dict(max_steps=max_steps,
                       max_seconds=max_seconds, val_max_batches=val_max_batches)), stream, indent=2)
    print(f"[resume] {provenance['mode']} source={provenance['source']} optimizer_reset={provenance['optimizer_reset']}", flush=True)
    event("start", provenance=provenance)

    def budget_exhausted():
        return ((max_steps > 0 and state["global_step"] - invocation_step >= max_steps) or
                (max_seconds > 0 and time.monotonic() - started >= max_seconds))

    def save(name="last.pt"):
        state["loader_rng"] = loader_gen.get_state()
        save_checkpoint(out / name, model, cfg, opt, sched, scaler, state, provenance)

    def stop_nonfinite(reason, **fields):
        # last.pt remains the last known healthy state; failed.pt preserves diagnostics.
        state["stop_reason"] = reason
        event("nonfinite_stop", reason=reason, **fields)
        opt.zero_grad(set_to_none=True)
        save("failed.pt")
        raise FloatingPointError(f"{reason}: stopped before optimizer update; see train_events.jsonl and failed.pt")

    def record_amp_skip(scale_before, grad_norm):
        state["amp_skips"] += 1
        state["amp_consecutive_skips"] += 1
        opt.zero_grad(set_to_none=True)
        event("amp_overflow_skip", scale_before=scale_before, scale_after=scaler.get_scale(),
              grad_norm=str(float(grad_norm)), consecutive_skips=state["amp_consecutive_skips"],
              patience=amp_overflow_patience, lr=[group["lr"] for group in opt.param_groups])
        print(f"ep{state['epoch']} it{state['global_step']} AMP_SKIP "
              f"consecutive={state['amp_consecutive_skips']}/{amp_overflow_patience} scale={scaler.get_scale()}", flush=True)
        if state["amp_consecutive_skips"] >= amp_overflow_patience:
            stop_nonfinite("amp_overflow_limit", consecutive_skips=state["amp_consecutive_skips"],
                           patience=amp_overflow_patience, scale=scaler.get_scale())

    save()  # Always provide a healthy recovery point, including failures on the first batch.

    for ep in range(state["epoch"], t.epochs):
        model.train(); set_frozen_eval(model, t.stage)
        skip_batches = state["batch_in_epoch"]
        if resume_rng is not None and skip_batches:
            # Replay data only to reconstruct sample order, augmentation and worker RNGs.
            # In-flight prefetched samples are recreated from the epoch's worker seeds.
            restore_rng(state["epoch_rng"])
            loader_gen.set_state(state["epoch_loader_rng"])
        else:
            state["epoch_rng"] = rng_state()
            state["epoch_loader_rng"] = loader_gen.get_state()
        iterator = iter(dl)
        for _ in range(skip_batches):
            next(iterator)
        if resume_rng is not None:
            restore_rng(resume_rng)
            resume_rng = None
        state["epoch"] = ep
        while state["batch_in_epoch"] < len(dl):
            if budget_exhausted():
                save(); event("budget_stop")
                print(f"[budget] stopped at epoch={ep} batch={state['batch_in_epoch']} global_step={state['global_step']}", flush=True)
                return
            batch = next(iterator)
            state["batch_in_epoch"] += 1
            step = state["global_step"]
            p_pred = min(1.0, step / max(1.0, t.pred_geom_warmup * total))
            # Pin the precision explicitly; implicit CPU AMP defaults to bf16, which
            # torchvision deform_conv2d does not support on this installation.
            with torch.autocast(dev, enabled=t.amp, dtype=amp_dtype):
                losses = model.forward_train(batch, pred_geom_prob=p_pred)
            if not losses:
                state["empty_skips"] += 1
                print(f"ep{ep} it{step} SKIP empty-loss batch", flush=True)
                continue
            state["loss_batches"] += 1; state["cic_batches"] += "cic" in losses
            loss = sum(losses.values())
            if not torch.isfinite(loss.detach()).all():
                stop_nonfinite("nonfinite_loss", losses={key: str(float(value.detach())) for key, value in losses.items()})
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 5.0)
            scale_before = scaler.get_scale()
            if not torch.isfinite(grad_norm):
                if not scaler.is_enabled():
                    stop_nonfinite("nonfinite_gradient", grad_norm=str(float(grad_norm)), loss=float(loss.detach()))
                # No optimizer step occurs. Explicit backoff also handles finite elementwise
                # gradients whose aggregate norm overflows during clipping.
                scaler.update(new_scale=scale_before * scaler.get_backoff_factor())
                record_amp_skip(scale_before, grad_norm)
                continue
            scaler.step(opt); scaler.update()
            if scaler.is_enabled() and scaler.get_scale() < scale_before:
                record_amp_skip(scale_before, grad_norm)
                continue
            state["amp_consecutive_skips"] = 0
            sched.step(); state["global_step"] += 1
            step = state["global_step"]
            if step % 20 == 0 or step - invocation_step == 1:
                values = {key: float(value.detach()) for key, value in losses.items()}
                event("update", loss=float(loss.detach()), losses=values, lr=[g["lr"] for g in opt.param_groups])
                print(f"ep{ep} it{step} loss {loss.item():.3f} " + " ".join(f"{k}={v:.3f}" for k, v in values.items()), flush=True)
            if checkpoint_every > 0 and step % checkpoint_every == 0:
                save()
        # A budget hit at the final batch leaves validation pending for a later exact resume.
        if budget_exhausted():
            save(); event("budget_stop", validation_pending=bool(val_dl is not None and (ep + 1) % val_every == 0))
            return
        if val_dl is not None and (ep + 1) % val_every == 0:
            model.eval(); ev = SpottingEvaluator(); evaluated = 0
            with torch.no_grad():
                for batch_index, vb in enumerate(val_dl):
                    if (val_max_batches > 0 and evaluated >= val_max_batches) or budget_exhausted():
                        break
                    vres = model.infer(vb["image"].to(dev))
                    for b, preds in enumerate(vres):
                        s, (ox, oy) = vb["scale"][b], vb["offset"][b]
                        for p in preds:
                            p["poly"] = (np.asarray(p["poly"]) - [ox, oy]) / s
                        ev.add(preds, vb["gt_orig"][b], image_bin=vb["meta"][b].get("illum_bin"))
                    evaluated += 1
            partial = evaluated < len(val_dl)
            f1 = ev.summary()["all"]["e2e_F"] if evaluated else None
            event("validation", e2e_F=f1, batches=evaluated, total_batches=len(val_dl), partial=partial,
                  condition=getattr(t, "val_condition", "original"))
            print(f"[val] ep{ep} e2e_F={f1} batches={evaluated}/{len(val_dl)} partial={partial}", flush=True)
            if f1 is not None and not partial and f1 > state["best_f1"]:
                state["best_f1"] = f1
                state["val_no_improve"] = 0
                save("best.pt")
            elif f1 is not None and not partial:
                state["val_no_improve"] += 1
            if budget_exhausted():
                # Keep completed-epoch data position; exact resume can run a full validation.
                save(); event("budget_stop", validation_pending=True)
                return
        state["epoch"] = ep + 1; state["batch_in_epoch"] = 0
        state.pop("epoch_rng", None); state.pop("epoch_loader_rng", None)
        save(); event("epoch_end")
        print(f"[train] ep{ep} done: updates={state['global_step']} empty_skips={state['empty_skips']} amp_skips={state['amp_skips']}", flush=True)
        if early_stop_patience > 0 and state["val_no_improve"] >= early_stop_patience:
            state["stop_reason"] = "early_stop"
            save(); event("early_stop", patience=early_stop_patience, best_f1=state["best_f1"])
            print(f"[early_stop] {state['val_no_improve']} complete validations without improvement", flush=True)
            return
    event("complete")


if __name__ == "__main__":
    main()
