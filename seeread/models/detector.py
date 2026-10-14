"""Detector following *Seeing Text in the Dark*: ResNet50 + Dynamic-Snake FPN (DSF) + train-only SCM.
Deviations from the paper (all flagged, ablate them):
  * SnakeConvLite replaces the official DSConv (swap in YaoleiQi/DSCNet for exact reproduction).
  * Gated self-attention is *channel-wise* (O(C^2)) -- token-wise N x N attention at 1/4 resolution is too costly.
  * A residual connection is added around the attention gate.
  * Direction is regressed as (cos, sin) instead of a raw angle (avoids wrap-around).
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F, torchvision
from torchvision.ops import deform_conv2d


def cbr(cin, cout, k=3):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True))


class SnakeConvLite(nn.Module):
    """Axis-wise snake convolution: horizontal (1,k) and vertical (k,1) kernels whose taps are displaced
    perpendicular to the kernel axis by *cumulative* learned offsets anchored at the centre tap."""
    def __init__(self, cin, cout, k=9, scope=1.0):
        super().__init__()
        self.k, self.scope = k, scope
        self.off_h, self.off_v = nn.Conv2d(cin, k, 3, padding=1), nn.Conv2d(cin, k, 3, padding=1)
        for m in (self.off_h, self.off_v):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        self.wh = nn.Parameter(torch.empty(cout, cin, 1, k)); self.wv = nn.Parameter(torch.empty(cout, cin, k, 1))
        nn.init.kaiming_normal_(self.wh); nn.init.kaiming_normal_(self.wv)
        self.fuse = nn.Sequential(nn.Conv2d(2 * cout, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True))

    def _accum(self, d):
        c = self.k // 2
        cum = torch.cumsum(torch.tanh(d) * self.scope, dim=1)
        return cum - cum[:, c:c + 1]

    def forward(self, x):
        oy, ox = self._accum(self.off_h(x)), self._accum(self.off_v(x))
        z = torch.zeros_like(oy)
        # torchvision offset layout: (B, 2*kh*kw, H, W) = [dy_0, dx_0, dy_1, dx_1, ...]
        off_h = torch.stack([oy, z], 2).flatten(1, 2)
        off_v = torch.stack([z, ox], 2).flatten(1, 2)
        yh = deform_conv2d(x, off_h, self.wh, padding=(0, self.k // 2))
        yv = deform_conv2d(x, off_v, self.wv, padding=(self.k // 2, 0))
        return self.fuse(torch.cat([yh, yv], 1))


class PMBlock(nn.Module):
    """Perceptual modulation block: [Conv || DSC] -> gated self-attention (Eq. 3-5)."""
    def __init__(self, cin, cout, heads=4):
        super().__init__()
        self.conv, self.dsc = cbr(cin, cout), SnakeConvLite(cin, cout)
        self.proj = nn.Conv2d(2 * cout, cout, 1)
        self.q, self.k = nn.Conv2d(cout, cout, 1), nn.Conv2d(cout, cout, 1)
        self.heads, self.temp = heads, nn.Parameter(torch.ones(heads, 1, 1))

    def forward(self, x):
        v = self.proj(torch.cat([self.conv(x), self.dsc(x)], 1))
        B, C, H, W = v.shape
        sh = lambda t: t.reshape(B, self.heads, C // self.heads, H * W)
        q, k = F.normalize(sh(F.gelu(self.q(v))), dim=-1), F.normalize(sh(F.gelu(self.k(v))), dim=-1)
        attn = ((q @ k.transpose(-2, -1)) * self.temp).softmax(-1)
        return (attn @ sh(v)).reshape(B, C, H, W) + v


class DSF(nn.Module):
    def __init__(self, in_chs=(256, 512, 1024, 2048), ch=128):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c, ch, 1) for c in in_chs])
        self.top = PMBlock(ch, ch)
        self.blocks = nn.ModuleList([PMBlock(2 * ch, ch) for _ in range(3)])

    def forward(self, feats):
        l = [m(f) for m, f in zip(self.lat, feats)]
        f = self.top(l[3])
        for j, i in enumerate((2, 1, 0)):
            f = F.interpolate(f, size=l[i].shape[-2:], mode="bilinear", align_corners=False)
            f = self.blocks[j](torch.cat([l[i], f], 1))
        return f  # stride 4


class SCM(nn.Module):
    """Spatial-constrained learning module (TRAIN ONLY): reconstructs the text-position mask from the deepest
    feature + positional embedding (L_sr) and exposes a semantic map to align the main branch with (L_ss)."""
    def __init__(self, c_in=2048, ch=128):
        super().__init__()
        self.reduce, self.pos = nn.Conv2d(c_in, ch, 1), nn.Conv2d(2, ch, 1)
        self.dec = nn.Sequential(cbr(ch, ch), cbr(ch, ch))
        self.mask_head, self.sem = nn.Conv2d(ch, 1, 1), nn.Conv2d(ch, ch, 1)

    def forward(self, c5, size):
        x = F.interpolate(self.reduce(c5), size=size, mode="bilinear", align_corners=False)
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, size[0], device=x.device),
                                torch.linspace(-1, 1, size[1], device=x.device), indexing="ij")
        x = self.dec(x + self.pos(torch.stack([xs, ys])[None].expand(x.size(0), -1, -1, -1)))
        return self.mask_head(x), self.sem(x)


class SeeReadDetector(nn.Module):
    def __init__(self, ch=128, pretrained=True, ctx_dim=128):
        super().__init__()
        w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        r = torchvision.models.resnet50(weights=w)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layers = nn.ModuleList([r.layer1, r.layer2, r.layer3, r.layer4])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.dsf, self.scm = DSF(ch=ch), SCM(ch=ch)
        head = lambda o: nn.Sequential(cbr(ch, ch), nn.Conv2d(ch, o, 1))
        self.h_text, self.h_center, self.h_geo = head(1), head(1), head(3)  # geo = (log1p height, cos, sin)
        self.ctx_g = nn.Linear(2048, ctx_dim)
        self.ctx_mlp = nn.Sequential(nn.Linear(ctx_dim + 4, ctx_dim), nn.ReLU(True), nn.Linear(ctx_dim, ctx_dim))

    def forward(self, img, with_scm=False):
        x = self.stem((img - self.mean) / self.std)
        feats = []
        for l in self.layers:
            x = l(x); feats.append(x)
        f = self.dsf(feats)
        out = dict(text=self.h_text(f), center=self.h_center(f), geo=self.h_geo(f), feat=f, c5=feats[-1])
        if with_scm:
            out["scm_mask"], out["scm_sem"] = self.scm(feats[-1], f.shape[-2:])
        return out

    def illum_context(self, img, c5):
        """Global illumination embedding shared with the enhancer: pooled deep feature + 4 luminance statistics."""
        y = (0.2126 * img[:, 0] + 0.7152 * img[:, 1] + 0.0722 * img[:, 2]).unsqueeze(1)
        y = F.adaptive_avg_pool2d(y, 16).flatten(1)
        stats = torch.stack([y.mean(1), y.std(1), y.amax(1), y.amin(1)], 1) * 10.0
        return self.ctx_mlp(torch.cat([self.ctx_g(c5.mean((2, 3))), stats], 1))
