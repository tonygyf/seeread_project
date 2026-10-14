"""SeeRead: See (detector w/ SCM+DSF+TSR) and Read (enhancer + recognizer) in ONE model, coupled through
  (1) a shared illumination context  (detector -> enhancer, gradients flow back into the detector backbone),
  (2) differentiable centerline strip rectification (recognition loss -> detector height head),
  (3) a train-only cross-illumination consistency loss (dark crop features ~ lit crop features)."""
from __future__ import annotations
import random
import numpy as np, torch, torch.nn as nn
from .detector import SeeReadDetector
from .enhancer import IlluminationAwareEnhancer, enhancer_regulariser
from .recognizer import BaseRecognizer
from .rectify import rectify_strip, sample_map
from .tsr import tsr_decode
from ..losses import detection_loss, cic_loss


class SeeRead(nn.Module):
    def __init__(self, recognizer: BaseRecognizer, cfg):
        super().__init__()
        m = cfg.model
        self.cfg, self.stride, self.crop_hw = cfg, 4, tuple(m.crop_hw)
        self.det = SeeReadDetector(ch=m.det_ch, pretrained=m.pretrained, ctx_dim=m.ctx_dim)
        self.enh = IlluminationAwareEnhancer(ctx_dim=m.ctx_dim, iters=m.enh_iters, use_ctx=m.use_ctx)
        self.rec = recognizer

    # ------------------------------------------------------------------ training
    def forward_train(self, batch, pred_geom_prob=0.0):
        c, w = self.cfg, self.cfg.loss
        dev = next(self.parameters()).device
        img = batch["image"].to(dev); gt = {k: v.to(dev) for k, v in batch["targets"].items()}
        lit = batch.get("lit"); lit = lit.to(dev) if lit is not None else None
        stage = c.train.stage
        losses = {}
        with torch.set_grad_enabled(stage in ("det", "joint")):
            out = self.det(img, with_scm=(stage != "rec" and c.model.use_scm))
            if stage != "rec":
                for k, v in detection_loss(out, gt).items():
                    losses[f"det_{k}"] = v * (w.det if k not in ("sr", "ss") else 1.0)
        if stage == "det":
            return losses
        ctx = self.det.illum_context(img, out["c5"])
        crops, crops_lit, texts, ctxs = [], [], [], []
        for b, inss in enumerate(batch["instances"]):
            for ins in inss:
                cen = torch.from_numpy(ins["centers"]).to(dev); hg = torch.from_numpy(ins["heights"]).to(dev)
                if random.random() < pred_geom_prob:       # curriculum: GT height -> predicted height
                    hp = torch.expm1(sample_map(out["geo"][b:b + 1, :1], cen, self.stride)[:, 0].clamp(min=0)) * self.stride
                    hg_use = hp
                else:
                    hg_use = hg
                crops.append(rectify_strip(img[b:b + 1], cen, hg_use, self.crop_hw))
                if lit is not None:
                    crops_lit.append(rectify_strip(lit[b:b + 1], cen, hg, self.crop_hw))
                texts.append(ins["text"]); ctxs.append(ctx[b:b + 1])
        if not crops:
            return losses
        x = torch.cat(crops); z = torch.cat(ctxs)
        y, prm = self.enh(x, z)
        losses["rec"] = w.rec * self.rec.forward_loss(y, texts)
        losses["enh"] = w.enh * enhancer_regulariser(x, y, prm)
        if crops_lit and w.cic > 0:
            with torch.no_grad():
                f_lit = self.rec.encode_features(torch.cat(crops_lit))
            losses["cic"] = w.cic * cic_loss(self.rec.encode_features(y), f_lit)
        return losses

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def infer(self, img, thr_center=0.5, thr_text=0.5, max_det=100):
        """img: (B,3,S,S) in [0,1]. Returns per-image list of dict(poly, score, text) in input-pixel coordinates."""
        self.eval()
        out = self.det(img)
        ctx = self.det.illum_context(img, out["c5"])
        tp, cp = torch.sigmoid(out["text"][:, 0]).cpu().numpy(), torch.sigmoid(out["center"][:, 0]).cpu().numpy()
        geo = torch.cat([out["geo"][:, :1].clamp(min=0), out["geo"][:, 1:]], 1).cpu().numpy()
        results = []
        for b in range(img.size(0)):
            dets = tsr_decode(tp[b], cp[b], geo[b], self.stride, thr_center, thr_text)[:max_det]
            crops = []
            for d in dets:
                if len(d["centers"]) < 2:
                    d["centers"] = np.concatenate([d["centers"] - [2, 0], d["centers"] + [2, 0]]); d["heights"] = np.repeat(d["heights"], 2)
                crops.append(rectify_strip(img[b:b + 1], torch.from_numpy(d["centers"]).float().to(img.device),
                                           torch.from_numpy(d["heights"]).float().to(img.device), self.crop_hw))
            if crops:
                x = torch.cat(crops)
                y, _ = self.enh(x, ctx[b:b + 1].expand(len(crops), -1))
                texts = self.rec.predict(y)
            else:
                texts = []
            results.append([dict(poly=d["poly"], score=d["score"], text=t) for d, t in zip(dets, texts)])
        return results
