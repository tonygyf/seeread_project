from __future__ import annotations
import torch, torch.nn.functional as F


def _dice(p, g, w):
    p, g = p * w, g * w
    return 1 - (2 * (p * g).sum((1, 2)) + 1) / (p.sum((1, 2)) + g.sum((1, 2)) + 1)


def detection_loss(out, gt, w_scm=(1.0, 1.0)):
    """L_seg (BCE+Dice on text/center) + L_H, L_dir (smooth-L1 on log-height and (cos,sin)) + L_sr (L1) + L_ss (L2)."""
    wgt = gt["weight"]
    L = {}
    for k in ("text", "center"):
        lg, g = out[k][:, 0], gt[k]
        bce = (F.binary_cross_entropy_with_logits(lg, g, reduction="none") * wgt).sum() / wgt.sum().clamp(min=1)
        L[f"seg_{k}"] = bce + _dice(torch.sigmoid(lg), g, wgt).mean()
    m = (gt["center"] > 0.5).float()
    n = m.sum().clamp(min=1)
    geo = out["geo"]
    L["h"] = (F.smooth_l1_loss(geo[:, 0], torch.log1p(gt["height"]), reduction="none") * m).sum() / n
    L["dir"] = ((F.smooth_l1_loss(geo[:, 1], gt["cos"], reduction="none") + F.smooth_l1_loss(geo[:, 2], gt["sin"], reduction="none")) * m).sum() / n
    if "scm_mask" in out:
        L["sr"] = w_scm[0] * F.l1_loss(torch.sigmoid(out["scm_mask"][:, 0]), gt["text"])
        L["ss"] = w_scm[1] * F.mse_loss(out["feat"], out["scm_sem"].detach())   # aux guides main (detach choice: ablate)
    return L


def cic_loss(f_dark, f_lit):
    """Cross-illumination consistency: token-wise 1 - cos(f_dark, sg[f_lit])."""
    return (1 - F.cosine_similarity(f_dark, f_lit.detach(), dim=-1)).mean()
