"""Illumination-Aware Enhancer (IAE): curve-based (SRDACE-style AAC) enhancement of text crops, FiLM-conditioned on a
GLOBAL illumination context taken from the detector. A tiny dark crop alone carries almost no information on how dark
the scene is -- the whole-image context from the detector is the 'bridge' between seeing and reading."""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


def aac(I, a, b):
    """Adaptive adjustment curve, Eq.(7) of Reading in the Dark. I in [0,1]; a, b per-pixel maps."""
    S = torch.sigmoid(-I + b - 0.1)
    return I + a * (1.0 / b) * S * I * (b - I)


class FiLMRes(nn.Module):
    def __init__(self, ch, ctx):
        super().__init__()
        self.c1, self.c2, self.film = nn.Conv2d(ch, ch, 3, padding=1), nn.Conv2d(ch, ch, 3, padding=1), nn.Linear(ctx, 2 * ch)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def forward(self, x, ctx):
        g, b = self.film(ctx).chunk(2, -1)
        h = F.relu(self.c1(x)) * (1 + g[..., None, None]) + b[..., None, None]
        return x + self.c2(h)


class IlluminationAwareEnhancer(nn.Module):
    def __init__(self, ctx_dim=128, ch=32, iters=6, alpha_max=3.0, n_blocks=3, use_ctx=True):
        super().__init__()
        self.iters, self.alpha_max, self.use_ctx = iters, alpha_max, use_ctx
        self.stem = nn.Conv2d(3, ch, 3, padding=1)
        self.blocks = nn.ModuleList([FiLMRes(ch, ctx_dim) for _ in range(n_blocks)])
        self.head = nn.Conv2d(ch, 2 * iters, 3, padding=1)
        nn.init.zeros_(self.head.weight); nn.init.constant_(self.head.bias, 0.0)

    def forward(self, x, ctx):
        """x: (N,3,H,W) in [0,1]; ctx: (N,ctx_dim). Returns enhanced crop and curve params."""
        if not self.use_ctx:
            ctx = torch.zeros_like(ctx)
        h = F.relu(self.stem(x))
        for blk in self.blocks:
            h = blk(h, ctx)
        p = self.head(h)
        a = self.alpha_max * torch.sigmoid(p[:, :self.iters])
        b = 0.2 + 0.8 * torch.sigmoid(p[:, self.iters:])
        y = x
        for i in range(self.iters):
            y = aac(y, a[:, i:i + 1], b[:, i:i + 1])
        return y.clamp(0, 1), dict(alpha=a, beta=b)


def sobel(x):
    gray = x.mean(1, keepdim=True)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    return torch.cat([F.conv2d(gray, kx, padding=1), F.conv2d(gray, kx.transpose(2, 3), padding=1)], 1)


def enhancer_regulariser(x_in, x_out, params, w_exp=1.0, w_col=0.5, w_tv=0.5, w_edge=1.0, target=0.5):
    """Light-weight, label-free regularisers (exposure / colour constancy / curve smoothness / Sobel edge content)."""
    exp = (F.avg_pool2d(x_out.mean(1, keepdim=True), 8) - target).abs().mean()
    m = x_out.mean((2, 3)); col = ((m[:, 0] - m[:, 1]) ** 2 + (m[:, 1] - m[:, 2]) ** 2 + (m[:, 0] - m[:, 2]) ** 2).mean()
    tv = sum(((p[..., 1:, :] - p[..., :-1, :]) ** 2).mean() + ((p[..., :, 1:] - p[..., :, :-1]) ** 2).mean()
             for p in (params["alpha"], params["beta"]))
    # edge content: match gradient *direction/structure*, scale-invariant (input is much darker than output)
    e_in, e_out = sobel(x_in), sobel(x_out)
    edge = (F.normalize(e_in.flatten(1), dim=1) - F.normalize(e_out.flatten(1), dim=1)).pow(2).sum(1).mean()
    return w_exp * exp + w_col * col + w_tv * tv + w_edge * edge
