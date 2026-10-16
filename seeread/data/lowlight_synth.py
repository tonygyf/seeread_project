"""Low-light synthesis.

Two modes:
  * "legacy"   -- the 6-step pipeline of *Reading in the Dark* (SSR -> k*I -> I^gamma -> AWGN -> vignette -> blur).
  * "physical" -- optics (blur, vignette) -> sensor (Poisson shot noise + Gaussian read noise) -> ISP gamma.
                  Signal-dependent noise is far closer to real night photos than the AWGN of "legacy".

IMPORTANT: TEST sets must be generated OFFLINE with fixed seeds (tools/make_synth_dark.py);
online synthesis (rng=None) is for TRAIN only.
"""
from __future__ import annotations
import numpy as np, cv2

# physical-mode exposure ratios (linear light) per level; calibrate against real LATeD luminance histogram!
PHYS_LEVELS = {"mild": (0.02, 0.04), "dark": (0.004, 0.01), "extreme": (0.0006, 0.0015)}
# legacy-mode (k range, gamma range, noise-sigma range in [0,1] units)
LEGACY_LEVELS = {"mild": ((0.5, 0.8), (1.5, 2.5), (0.005, 0.015)),
                 "dark": ((0.25, 0.5), (2.5, 3.5), (0.01, 0.03)),
                 "extreme": ((0.1, 0.3), (3.5, 5.0), (0.02, 0.05))}
# illumination bins on mean luminance Y of the *final* image (re-calibrate on real data)
BIN_EDGES = (0.0, 0.04, 0.08, 0.15, 1.01)
BIN_NAMES = ("extreme", "dark", "dim", "lit")


def luminance(img01: np.ndarray) -> float:
    y = 0.2126 * img01[..., 0] + 0.7152 * img01[..., 1] + 0.0722 * img01[..., 2]
    return float(y.mean())


def illum_bin(y: float) -> str:
    for i in range(len(BIN_NAMES)):
        if BIN_EDGES[i] <= y < BIN_EDGES[i + 1]:
            return BIN_NAMES[i]
    return BIN_NAMES[-1]


def single_scale_retinex(img01: np.ndarray, sigma: float = 80.0) -> np.ndarray:
    sigma = min(sigma, 0.3 * max(img01.shape[:2]))
    x = img01.astype(np.float32) * 255.0 + 1.0
    r = np.log(x) - np.log(cv2.GaussianBlur(x, (0, 0), max(sigma, 1.0)))
    return (r - r.min()) / (r.max() - r.min() + 1e-6)


def _vignette(h, w, rng, strength=(0.4, 0.8)):
    s = rng.uniform(0.6, 1.2)
    xs = np.exp(-((np.arange(w) - w / 2) ** 2) / (2 * (s * w) ** 2))
    ys = np.exp(-((np.arange(h) - h / 2) ** 2) / (2 * (s * h) ** 2))
    v = ys[:, None] * xs[None, :]
    v = v / v.max()
    corner = v[0, 0]
    tgt = rng.uniform(*strength)
    if corner < tgt:  # keep vignetting mild enough
        v = tgt + (1 - tgt) * (v - corner) / (1 - corner + 1e-6)
    return v.astype(np.float32)[..., None]


def darken_legacy(img_u8, rng, level="dark", blur_p=0.3, use_ssr=True):
    x = img_u8.astype(np.float32) / 255.0
    if use_ssr:
        x = single_scale_retinex(x)
    (k0, k1), (g0, g1), (n0, n1) = LEGACY_LEVELS[level]
    k, g, sig = rng.uniform(k0, k1), rng.uniform(g0, g1), rng.uniform(n0, n1)
    x = (k * x) ** g
    x = np.clip(x + rng.normal(0, sig, x.shape), 0, 1)
    x = x * _vignette(*x.shape[:2], rng)
    blur = rng.random() < blur_p
    if blur:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 1.5))
    return (x * 255).astype(np.uint8), dict(mode="legacy", level=level, k=k, gamma=g, sigma=sig, blur=blur)


def darken_physical(img_u8, rng, level="dark", blur_p=0.3, peak=(800.0, 4000.0), read=(0.002, 0.01)):
    x = img_u8.astype(np.float32) / 255.0
    lin = np.power(x, 2.2)
    ratio = float(np.exp(rng.uniform(*np.log(PHYS_LEVELS[level]))))
    sig = lin * ratio
    blur = rng.random() < blur_p
    if blur:  # optical / motion blur happens BEFORE the sensor
        sig = cv2.GaussianBlur(sig, (0, 0), rng.uniform(0.5, 1.5))
    sig = sig * _vignette(*sig.shape[:2], rng)
    pk, rd = rng.uniform(*peak), rng.uniform(*read)
    noisy = rng.poisson(np.clip(sig, 0, None) * pk).astype(np.float32) / pk + rng.normal(0, rd, sig.shape)
    out = np.power(np.clip(noisy, 0, 1), 1 / 2.2)
    return (out * 255).astype(np.uint8), dict(mode="physical", level=level, ratio=ratio, peak=pk, read=rd, blur=blur)


def darken(img_u8, rng=None, level=None, mode="physical", **kw):
    """img_u8: HxWx3 RGB uint8. Returns (dark_u8, meta). meta['illum_y'] / meta['illum_bin'] are filled."""
    rng = rng if rng is not None else np.random.default_rng()
    level = level or rng.choice(["mild", "dark"])   # extreme dropped: exposure ratio 0.06% destroys text texture irreversibly
    fn = darken_physical if mode == "physical" else darken_legacy
    out, meta = fn(img_u8, rng, level, **kw)
    meta["illum_y"] = luminance(out.astype(np.float32) / 255.0)
    meta["illum_bin"] = illum_bin(meta["illum_y"])
    return out, meta
