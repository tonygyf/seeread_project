"""Usage: python tools/train.py --config configs/seeread_joint.yaml [--set train.epochs=1]
Stages: det (detector only) -> rec (enhancer+recognizer on GT geometry, detector frozen) -> joint (all)."""
import argparse, os, random, sys, json
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--set", nargs="*")
    a = ap.parse_args(); cfg = load_config(a.config, parse_set(a.set)); t = cfg.train
    random.seed(t.seed); np.random.seed(t.seed); torch.manual_seed(t.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SeeRead(build_recognizer(cfg.rec), cfg).to(dev)
    if t.init_det: model.det.load_state_dict(torch.load(t.init_det, map_location=dev)["det"])
    if t.init_rec:
        sd = torch.load(t.init_rec, map_location=dev); model.enh.load_state_dict(sd["enh"]); model.rec.load_state_dict(sd["rec"])
    if t.stage == "rec":
        for p in model.det.parameters(): p.requires_grad = False
    elif t.stage == "det":
        for m in (model.enh, model.rec):
            for p in m.parameters(): p.requires_grad = False
    if getattr(t, "freeze_rec", False) and t.stage == "joint":   # S3-v3: protect the S2 recognizer
        for p in model.rec.parameters(): p.requires_grad = False
    if getattr(cfg.data, "train_anns", None):               # multi-dataset mixture (S3-v3)
        from seeread.data.mixture import build_mixture
        ds, sampler, _mix_sizes = build_mixture(
            cfg.data.root, cfg.data.train_anns, cfg.data.mix_weights,
            num_samples=int(getattr(cfg.data, "mix_samples_per_epoch", 0) or 0) or None,
            train=True, size=cfg.data.size, synth_mode=cfg.data.synth_mode, seed=t.seed)
        print(f"[mix] dataset sizes={_mix_sizes} weights={cfg.data.mix_weights} draws/epoch={len(sampler)}", flush=True)
        dl = DataLoader(ds, t.batch_size, sampler=sampler, num_workers=cfg.data.workers,
                        collate_fn=collate, drop_last=True)
    else:
        ds = SeeReadDataset(cfg.data.root, cfg.data.train_ann, train=True, size=cfg.data.size, synth_mode=cfg.data.synth_mode, seed=t.seed)
        dl = DataLoader(ds, t.batch_size, shuffle=True, num_workers=cfg.data.workers, collate_fn=collate, drop_last=True)
    val_every = int(getattr(t, "val_every", 0) or 0)        # 0 disables validation; else eval every N epochs and keep best.pt
    val_dl = None
    if val_every > 0 and getattr(cfg.data, "val_ann", None):
        val_ds = SeeReadDataset(cfg.data.root, cfg.data.val_ann, train=False, size=cfg.data.size, synth_mode=cfg.data.synth_mode, seed=t.seed)
        val_dl = DataLoader(val_ds, 4, shuffle=False, num_workers=cfg.data.workers, collate_fn=collate)
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
    scaler = torch.amp.GradScaler(enabled=t.amp and dev == "cuda")
    os.makedirs(t.out_dir, exist_ok=True); step, total, n_skip, nb, ncic = 0, t.epochs * len(dl), 0, 0, 0

    def save_ckpt(path):
        torch.save(dict(det=model.det.state_dict(), enh=model.enh.state_dict(), rec=model.rec.state_dict(), cfg=vars(cfg.train)), path)

    best_f1 = -1.0
    for ep in range(t.epochs):
        model.train(); set_frozen_eval(model, t.stage)
        for batch in dl:
            p_pred = min(1.0, step / max(1.0, t.pred_geom_warmup * total))
            with torch.autocast(dev, enabled=t.amp):
                L = model.forward_train(batch, pred_geom_prob=p_pred)
            if not L:                      # e.g. rec stage with zero rec-eligible instances: skip update, schedule counts real steps only
                n_skip += 1
                print(f"ep{ep} it{step} SKIP empty-loss batch", flush=True)
                continue
            nb += 1; ncic += "cic" in L
            loss = sum(L.values())
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            if step % 20 == 0:
                lm = batch.get("lit_mask")
                pair = f" pair={int(lm.sum())}/{len(lm)}" if lm is not None else ""
                print(f"ep{ep} it{step} loss {loss.item():.3f}{pair} " + " ".join(f"{k}={v.item():.3f}" for k, v in L.items()), flush=True)
        save_ckpt(os.path.join(t.out_dir, "last.pt"))
        if nb:
            print(f"[train] ep{ep} done: {ncic}/{nb} batches with CIC, {n_skip} empty-loss skips so far", flush=True)
        if val_dl is not None and (ep + 1) % val_every == 0:
            model.eval(); ev = SpottingEvaluator()
            with torch.no_grad():
                for vb in val_dl:
                    vres = model.infer(vb["image"].to(dev))
                    for b, preds in enumerate(vres):              # back to original image coordinates
                        s, (ox, oy) = vb["scale"][b], vb["offset"][b]
                        for p in preds:
                            p["poly"] = (np.asarray(p["poly"]) - [ox, oy]) / s
                        ev.add(preds, vb["gt_orig"][b], image_bin=vb["meta"][b].get("illum_bin"))
            f1 = ev.summary()["all"]["e2e_F"]
            print(f"[val] ep{ep} e2e_F={f1:.4f}", flush=True)
            if f1 > best_f1:                                     # peak-epoch checkpoint on bright validation E2E F1
                best_f1 = f1
                save_ckpt(os.path.join(t.out_dir, "best.pt"))
                print(f"[val] ep{ep} new best {f1:.4f} -> best.pt", flush=True)
            model.train(); set_frozen_eval(model, t.stage)
    if n_skip:
        print(f"[train] {n_skip} batches skipped (empty loss) in total", flush=True)


if __name__ == "__main__":
    main()
