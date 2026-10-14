"""Usage: python tools/train.py --config configs/seeread_joint.yaml [--set train.epochs=1]
Stages: det (detector only) -> rec (enhancer+recognizer on GT geometry, detector frozen) -> joint (all)."""
import argparse, os, random, sys, json
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seeread.config import load_config
from seeread.data.datasets import SeeReadDataset, collate
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
    ds = SeeReadDataset(cfg.data.root, cfg.data.train_ann, train=True, size=cfg.data.size, synth_mode=cfg.data.synth_mode, seed=t.seed)
    dl = DataLoader(ds, t.batch_size, shuffle=True, num_workers=cfg.data.workers, collate_fn=collate, drop_last=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=t.lr, weight_decay=t.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, t.epochs * len(dl))
    scaler = torch.amp.GradScaler(enabled=t.amp and dev == "cuda")
    os.makedirs(t.out_dir, exist_ok=True); step, total = 0, t.epochs * len(dl)
    for ep in range(t.epochs):
        model.train()
        for batch in dl:
            p_pred = min(1.0, step / max(1.0, t.pred_geom_warmup * total))
            with torch.autocast(dev, enabled=t.amp):
                L = model.forward_train(batch, pred_geom_prob=p_pred)
            loss = sum(L.values())
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            if step % 20 == 0:
                print(f"ep{ep} it{step} loss {loss.item():.3f} " + " ".join(f"{k}={v.item():.3f}" for k, v in L.items()), flush=True)
        torch.save(dict(det=model.det.state_dict(), enh=model.enh.state_dict(), rec=model.rec.state_dict(), cfg=vars(cfg.train)),
                   os.path.join(t.out_dir, "last.pt"))


if __name__ == "__main__":
    main()
