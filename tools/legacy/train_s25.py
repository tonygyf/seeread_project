"""Stage S2.5: mixed real/synthetic recognition replay.

Runs AFTER S2 (frozen detector). Guards against the catastrophic forgetting observed when fine-tuning the enhancer/
recogniser on ESTR alone (a 2k-step pure-ESTR probe doubled real-dark word accuracy 7.8%->14.1% but collapsed
SynthText-domain accuracy 14.4%->4.3%). Each optimisation step mixes, at the CROP level:
  * (1-frac) SynthText strips  -- online mild/dark darkening, GT geometry, lit pair (synthetic replay, anti-forgetting)
  * frac     ESTR real crops   -- REAL dark/well-lit word-crop pairs with transcriptions (target-domain supervision)
Both sources supply a lit GT crop, so the enhancer L1 reconstruction loss (loss.enh_rec) gets REAL paired targets
on the ESTR fraction. Detector stays frozen; only enhancer + recogniser train.

Usage:
  python tools/legacy/train_s25.py --config configs/base.yaml \
    --init results/archive/legacy/v2/s2_rec_v2/last.pt --out results/s25_mix \
    --steps 1400 --bs 96 --estr-frac 0.25 --lr 1.5e-4
"""
import argparse, json, os, random, sys
import numpy as np, cv2, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from seeread.config import load_config
from seeread.data.datasets import SeeReadDataset, collate
from seeread.models.recognizer import build_recognizer
from seeread.models.seeread import SeeRead
from seeread.models.rectify import rectify_strip
from seeread.models.enhancer import enhancer_regulariser
from seeread.losses import cic_loss
from tools.legacy.train import parse_set

ESTR_ROOT = os.path.join("data", "raw", "ESTR")
CROP_HW = (64, 256)


def _estr_load(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (CROP_HW[1], CROP_HW[0]), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(im).permute(2, 0, 1).float() / 255.0


class ESTRCrops(Dataset):
    """REAL paired dark/lit word crops with transcriptions (data/raw/ESTR/{split}.jsonl)."""

    def __init__(self, split="train", root=ESTR_ROOT):
        rows = [json.loads(l) for l in open(os.path.join(root, f"{split}.jsonl"), encoding="utf-8")]
        self.root, self.rows = root, [r for r in rows if "".join(c for c in r["label"] if c.isalnum())]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return (_estr_load(os.path.join(self.root, r["low_light"])),
                _estr_load(os.path.join(self.root, r["well_lit"])), r["label"])


def estr_collate(b):
    return torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b]), [x[2] for x in b]


class SynthStripIter(IterableDataset):
    """Yields (dark_strip, lit_strip|None, text) from the full-image SeeRead pipeline with the FROZEN detector:
    reproduces the rec-stage crop extraction of SeeRead.forward_train (GT geometry only)."""

    def __init__(self, model, cfg, dev, img_bs):
        self.m, self.cfg, self.dev, self.img_bs = model, cfg, dev, img_bs

    def __iter__(self):
        ds = SeeReadDataset(self.cfg.data.root, self.cfg.data.train_ann, train=True,
                            size=self.cfg.data.size, seed=self.cfg.train.seed)
        dl = DataLoader(ds, self.img_bs, shuffle=True, num_workers=self.cfg.data.workers,
                        collate_fn=collate, drop_last=True)
        m, dev = self.m, self.dev
        while True:
            for batch in dl:
                img = batch["image"].to(dev)
                lit = batch.get("lit"); lit = lit.to(dev) if lit is not None else None
                lit_mask = batch.get("lit_mask")
                with torch.no_grad():
                    out = m.det(img)
                    ctx_all = m.det.illum_context(img, out["c5"])
                for b, inss in enumerate(batch["instances"]):
                    for ins in inss:                       # rec stage: only GT-geometry, transcription-valid instances
                        cen = torch.from_numpy(ins["centers"]).to(dev)
                        hg = torch.from_numpy(ins["heights"]).to(dev)
                        dark = rectify_strip(img[b:b + 1], cen, hg, m.crop_hw)
                        pair_lit = None
                        if lit is not None and bool(lit_mask[b]):
                            pair_lit = rectify_strip(lit[b:b + 1], cen, hg, m.crop_hw)
                        yield dark, pair_lit, ins["text"], ctx_all[b:b + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init", default=None, help="S2 checkpoint (det frozen, enh+rec initialised); random if unset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1400)
    ap.add_argument("--bs", type=int, default=96, help="total recognition crops per step")
    ap.add_argument("--estr-frac", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--img-bs", type=int, default=24, help="full images per SynthText detector pass")
    ap.add_argument("--estr-workers", type=int, default=2)
    ap.add_argument("--estr-root", default=ESTR_ROOT, help="root containing train.jsonl + low_light_train/well_lit_train")
    ap.add_argument("--set", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, parse_set(a.set)); t = cfg.train; w = cfg.loss
    random.seed(t.seed); np.random.seed(t.seed); torch.manual_seed(t.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SeeRead(build_recognizer(cfg.rec), cfg).to(dev)
    sd = torch.load(a.init, map_location=dev) if a.init else None
    if sd is not None:
        model.det.load_state_dict(sd["det"]); model.enh.load_state_dict(sd["enh"]); model.rec.load_state_dict(sd["rec"])
    for p in model.det.parameters():
        p.requires_grad = False
    params = list(model.enh.parameters()) + list(model.rec.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=t.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    scaler = torch.amp.GradScaler(enabled=t.amp and dev == "cuda")
    os.makedirs(a.out, exist_ok=True)

    estr = ESTRCrops(root=a.estr_root)
    estr_dl = DataLoader(estr, batch_size=max(1, int(round(a.bs * a.estr_frac))), shuffle=True,
                         num_workers=a.estr_workers, collate_fn=estr_collate, drop_last=True)
    n_estr = max(1, int(round(a.bs * a.estr_frac))); n_synth = a.bs - n_estr
    model.train(); model.det.eval()
    synth_it = iter(SynthStripIter(model, cfg, dev, a.img_bs))
    estr_it = iter(estr_dl)

    def synth_ctx(dev_):
        return next(synth_it)

    step = 0
    while step < a.steps:
        dark, lit_gt, texts, ctxs = [], [], [], []
        for _ in range(n_synth):
            d, lp, tx, cx = synth_ctx(dev)
            dark.append(d); texts.append(tx); ctxs.append(cx)
            if lp is not None: lit_gt.append((len(dark) - 1, lp))
        try:
            ed, el, et = next(estr_it)
        except StopIteration:
            estr_it = iter(estr_dl); ed, el, et = next(estr_it)
        ed, el = ed.to(dev), el.to(dev)
        with torch.no_grad():                       # ESTR crop is its own (small) image -> crop-derived context
            ex = model.det.stem((ed - model.det.mean) / model.det.std)
            for ly in model.det.layers:
                ex = ly(ex)
            ectx = model.det.illum_context(ed, ex)
        estr_idx = []
        for j in range(ed.size(0)):
            estr_idx.append((len(dark), el[j:j + 1])); dark.append(ed[j:j + 1]); texts.append(et[j]); ctxs.append(ectx[j:j + 1])
        x = torch.cat(dark); z = torch.cat(ctxs)
        pair_idx = [i for i, _ in lit_gt] + [i for i, _ in estr_idx]
        lit_list = [lp for _, lp in lit_gt] + [lp for _, lp in estr_idx]
        with torch.autocast(dev, enabled=t.amp):
            y, prm = model.enh(x, z)
            L = {}
            l_rec = model.rec.forward_loss(y, texts)
            if l_rec is None:
                continue
            L["rec"] = w.rec * l_rec
            L["enh"] = w.enh * enhancer_regulariser(x, y, prm)
            if pair_idx and getattr(w, "enh_rec", 0.0) > 0:
                L["enh_rec"] = w.enh_rec * F.l1_loss(y[torch.tensor(pair_idx, device=dev)], torch.cat(lit_list))
            if pair_idx and w.cic > 0:
                f_lit = model.rec.teacher_features(torch.cat(lit_list))
                L["cic"] = w.cic * cic_loss(model.rec.encode_features(y[torch.tensor(pair_idx, device=dev)]), f_lit)
            loss = sum(L.values())
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 5.0)
        scaler.step(opt); scaler.update(); sched.step(); step += 1
        if step % 50 == 0:
            print(f"it{step} loss {loss.item():.3f} " + " ".join(f"{k}={v.item():.3f}" for k, v in L.items())
                  + f" (synth={n_synth} estr={n_estr})", flush=True)
    torch.save(dict(det=model.det.state_dict(), enh=model.enh.state_dict(), rec=model.rec.state_dict(),
                    cfg=vars(cfg.train)), os.path.join(a.out, "last.pt"))
    print(f"S25_DONE saved {os.path.join(a.out, 'last.pt')}", flush=True)


if __name__ == "__main__":
    main()
