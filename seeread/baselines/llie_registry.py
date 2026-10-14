"""LLIE front-ends for the CASCADE baselines (enhance -> detect -> read). Built-ins need no extra deps;
external learned enhancers (ZeroDCE/SCI/RUAS/SRDACE/GDP) are adapters you fill in from the official repos."""
import cv2, numpy as np
from ..data.lowlight_synth import single_scale_retinex


def identity(img): return img
def gamma(img, g=0.4): return (255 * (img / 255.0) ** g).astype(np.uint8)
def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB); lab[..., 0] = cv2.createCLAHE(3.0, (8, 8)).apply(lab[..., 0]); return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
def ssr(img): return (255 * single_scale_retinex(img.astype(np.float32) / 255)).astype(np.uint8)


def _external(name):
    def f(img):
        raise NotImplementedError(f"wrap the official {name} inference here (RGB uint8 HxWx3 in/out)")
    return f


REGISTRY = dict(identity=identity, gamma=gamma, clahe=clahe, ssr=ssr, **{n: _external(n) for n in ("zerodce", "sci", "ruas", "srdace", "gdp")})
