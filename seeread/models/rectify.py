"""Centerline-anchored differentiable strip rectification (CSR).
Crop geometry = non-differentiable centerline points (FPS centres) + DIFFERENTIABLE heights sampled from the detector's
height map. Recognition loss therefore back-propagates into the detector's height head (and, through the illumination
context, into its backbone)."""
from __future__ import annotations
import torch, torch.nn.functional as F


def sample_map(m, pts_xy, stride):
    """m: (1,C,Hf,Wf); pts_xy: (K,2) input-px. Bilinear sample -> (K,C)."""
    Hf, Wf = m.shape[-2:]
    gx = (pts_xy[:, 0] / stride) / max(Wf - 1, 1) * 2 - 1
    gy = (pts_xy[:, 1] / stride) / max(Hf - 1, 1) * 2 - 1
    grid = torch.stack([gx, gy], -1)[None, None]
    return F.grid_sample(m, grid, mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0].t()


def rectify_strip(img, centers, heights, out_hw=(64, 256), pad=1.15):
    """img: (1,3,H,W); centers: (K,2) px (K>=2); heights: (K,) px (may carry grad). Returns (1,3,Ho,Wo)."""
    Ho, Wo = out_hw
    K = centers.shape[0]
    d = centers[1:] - centers[:-1]
    t = torch.cat([d[:1], 0.5 * (d[1:] + d[:-1]), d[-1:]], 0) if K > 2 else torch.cat([d, d], 0)
    t = t / t.norm(dim=1, keepdim=True).clamp(min=1e-6)
    nrm = torch.stack([-t[:, 1], t[:, 0]], -1)                      # points "down" for left->right text
    hh = (heights.clamp(min=2.0) * pad)[:, None] / 2
    top, bot = centers - nrm * hh, centers + nrm * hh
    seg = (centers[1:] - centers[:-1]).norm(dim=1)
    s = torch.cat([seg.new_zeros(1), seg.cumsum(0)]).detach()
    s = s / s[-1].clamp(min=1e-6)
    u = torch.linspace(0, 1, Wo, device=img.device)
    idx = torch.searchsorted(s.contiguous(), u, right=True).clamp(1, K - 1)
    w = ((u - s[idx - 1]) / (s[idx] - s[idx - 1]).clamp(min=1e-6)).clamp(0, 1)[:, None]
    top_u, bot_u = top[idx - 1] * (1 - w) + top[idx] * w, bot[idx - 1] * (1 - w) + bot[idx] * w
    v = torch.linspace(0, 1, Ho, device=img.device)[:, None, None]
    px = top_u[None] * (1 - v) + bot_u[None] * v                    # (Ho,Wo,2)
    H, W = img.shape[-2:]
    grid = torch.stack([px[..., 0] / (W - 1) * 2 - 1, px[..., 1] / (H - 1) * 2 - 1], -1)[None]
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)
