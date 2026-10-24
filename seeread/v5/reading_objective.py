"""Reading Eq.(12–15), with explicitly pinned Self-DACE LLE components.

Reading specifies phi=1e5 and omega=100, but not all LLE implementation
details. We retain the original stage1 loss definitions and weights, apply
them to the final enhanced image, and register the Sobel convention below.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from seeread.v5.enhancers import SelfDACEEnhancer, _load_model_source
from seeread.v5.recognition import ReadingPipeline

LOSS_SOURCE = 'codes_SelfDACE/old_version/stage1/Myloss.py'
LOSS_SHA256 = '8fa220c3ef8064f55131579c4647d47e4e9a589ec67922ba704f83a050b4fa25'


def sobel_magnitude(image):
    """Mean-RGB grayscale, raw Sobel kernels, reflect borders, eps=1e-12."""
    gray = image.mean(1, keepdim=True)
    kernel = image.new_tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]])[:, None]
    gradient = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='reflect'), kernel)
    return (gradient.square().sum(1, keepdim=True) + 1e-12).sqrt()


class ReadingJointObjective(nn.Module):
    def __init__(self, source_root, edge_weight=1e5, ocr_weight=100.,
                 local_weight=1000., global_weight=1500., brightness_weight=5.,
                 alpha_weight=1000., beta_weight=5000.):
        super().__init__()
        path = Path(source_root) / LOSS_SOURCE
        if hashlib.sha256(path.read_bytes()).hexdigest() != LOSS_SHA256:
            raise ValueError('Self-DACE loss source differs from the pinned public implementation')
        source = _load_model_source(path)
        self.local = source.L_localcolor()
        self.global_color = source.L_color()
        self.brightness = source.L_exp(4, .8)
        self.curve = source.L_SMO()
        self.weights = dict(edge=float(edge_weight), ocr=float(ocr_weight), local=float(local_weight),
                            global_color=float(global_weight), brightness=float(brightness_weight),
                            alpha=float(alpha_weight), beta=float(beta_weight))
        if any(w < 0 for w in self.weights.values()):
            raise ValueError('Reading objective weights must be nonnegative')
        self.provenance = dict(
            paper='https://arxiv.org/html/2604.23685v1#S3.SS4',
            equations='12–15', loss_source=str(path), source_sha256=LOSS_SHA256,
            weights=self.weights,
            assumptions=['Self-DACE stage1 published LLE definitions/weights map to Reading LCC/GCF/BP/CR',
                         'LLE evaluated on final denoised/clamped enhancer output',
                         'Sobel uses mean-RGB grayscale, unnormalized kernels, reflect border, magnitude eps1e-12',
                         'paper does not specify these LLE and Sobel implementation choices'],
        )

    def enhancement_terms(self, low, enhanced, alpha, beta):
        # Called in FP32: phi=1e5 can overflow half precision and should not
        # silently change the published objective through AMP loss overflow.
        raw = dict(lcc=self.local(enhanced, low).mean(), gcf=self.global_color(enhanced).mean(),
                   bp=self.brightness(enhanced, low).mean(), cr_alpha=self.curve(alpha), cr_beta=self.curve(beta),
                   edge=(sobel_magnitude(enhanced) - sobel_magnitude(low)).square().mean())
        w = self.weights
        lle = (w['local'] * raw['lcc'] + w['global_color'] * raw['gcf'] + w['brightness'] * raw['bp'] +
               w['alpha'] * raw['cr_alpha'] + w['beta'] * raw['cr_beta'])
        return dict(**raw, lle=lle, llie=lle + w['edge'] * raw['edge'])


class PaperReadingPipeline(ReadingPipeline):
    def __init__(self, enhancer, recognizer, objective):
        if not isinstance(enhancer, SelfDACEEnhancer):
            raise ValueError('This Reading LLE adapter requires official Self-DACE with alpha/beta curves')
        super().__init__(enhancer, recognizer)
        self.objective = objective
        self.enhancer.provenance = dict(self.enhancer.provenance, training_objective=objective.provenance)
        self.last_loss_components = {}

    def forward_loss(self, crops, texts):
        with torch.autocast(device_type=crops.device.type, enabled=False):
            low = crops.float()
            light, alpha, beta = self.enhancer.light(low)
            enhanced = self.enhancer.denoise(light) if self.enhancer.denoise is not None else light
            if self.enhancer.clamp_output:
                enhanced = enhanced.clamp(0, 1)
            terms = self.objective.enhancement_terms(low, enhanced, alpha.float(), beta.float())
        ocr = self.recognizer.forward_loss(enhanced, texts)
        total = terms['llie'] + self.objective.weights['ocr'] * ocr.float()
        self.last_loss_components = {name: float(value.detach()) for name, value in terms.items()}
        self.last_loss_components.update(ocr=float(ocr.detach()), total=float(total.detach()))
        return total
