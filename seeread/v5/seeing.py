"""Seeing-style reconstruction with explicit architecture assumptions.

This is an independent paper-guided implementation, not an author checkpoint.
SCM is training-only: ground-truth masks never enter detection predictions.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
import torch.nn.functional as F


def cbr(cin, cout, kernel=3):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, padding=kernel // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=False))


class PerceptualModulation(nn.Module):
    """Eq.(4–5), channel-layout assumption, no learned temperature/residual.

    Conv and two-axis DSC produce C channels each. Concatenated V has 2C
    channels; attention preserves 2C, then a declared 1x1 shape adapter emits C.
    Q/K activation=ReLU and attention axis=channel are reconstruction choices.
    """

    def __init__(self, cin, channels, source_root, kernel_size=9, dsconv_factory=None):
        super().__init__()
        if dsconv_factory is None:
            from seeread.v5.dsconv import OfficialDSConv
            dsconv_factory = OfficialDSConv
        self.conv = cbr(cin, channels)
        self.snake_h = dsconv_factory(cin, channels, kernel_size=kernel_size,
                                     morph=0, source_root=source_root)
        self.snake_v = dsconv_factory(cin, channels, kernel_size=kernel_size,
                                     morph=1, source_root=source_root)
        self.snake_projection = nn.Conv2d(2 * channels, channels, 1)
        self.q = nn.Conv2d(2 * channels, 2 * channels, 1)
        self.k = nn.Conv2d(2 * channels, 2 * channels, 1)
        self.output_projection = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, x):
        snake = self.snake_projection(torch.cat((self.snake_h(x), self.snake_v(x)), 1))
        # The ordinary convolution can overflow FP16 before BN normalizes it
        # (real SynthText step85672). Retain AMP after this normalized branch.
        with torch.autocast(x.device.type, enabled=False):
            ordinary = self.conv(x.float())
        value = torch.cat((ordinary.to(snake.dtype), snake), 1)
        batch, channels, height, width = value.shape
        q = F.relu(self.q(value)).flatten(2)
        k = F.relu(self.k(value)).flatten(2)
        # FP16 can overflow QK before the sqrt(N) division at 640px.
        # Keep the paper expression; accumulate attention in FP32 under AMP.
        with torch.autocast(x.device.type, enabled=False):
            attention = torch.softmax((q.float() @ k.float().transpose(-2, -1)) / math.sqrt(height * width), dim=-1)
            modulated = (attention @ value.float().flatten(2)).reshape(batch, channels, height, width)
        modulated = modulated.to(value.dtype)
        return self.output_projection(modulated)


class SpatialConstraint(nn.Module):
    """Training-only mask encoder/decoder, with an explicit stop-grad target."""

    def __init__(self, channels):
        super().__init__()
        self.encoder = cbr(2, channels)
        self.decoder = nn.Sequential(cbr(channels, channels), cbr(channels, channels))
        self.reconstruction = nn.Conv2d(channels, 2, 1)

    def forward(self, image_feature, mask):
        if mask.ndim == 3:
            mask = F.one_hot(mask.long(), 2).permute(0, 3, 1, 2).float()
        if mask.ndim != 4 or mask.shape[1] != 2:
            raise ValueError('SCM requires a binary B,H,W mask or two-channel B,2,H,W target')
        position = self.encoder(mask.to(image_feature))
        position = F.interpolate(position, size=image_feature.shape[-2:], mode='bilinear', align_corners=False)
        auxiliary = self.decoder(image_feature + position)
        return dict(scm_mask=torch.sigmoid(self.reconstruction(auxiliary)), scm_sem=auxiliary)


class SeeingDetector(nn.Module):
    """ResNet50 → three 256-channel PM stages → two-class text/center heads.

    Named output geometry: positive *full* height in feature pixels and tangent
    angle in radians modulo pi. The decoder is responsible for one stride
    conversion. Geometry parameterization and branch architecture are explicit
    assumptions; see docs/04_v5_alignment.md.
    """

    def __init__(self, channels=256, source_root='/opt/dscnet', kernel_size=9,
                 pretrained=True, dsconv_factory=None, backbone_factory=None):
        super().__init__()
        if channels < 4:
            raise ValueError('Seeing channels must be at least four')
        if backbone_factory is None:
            from torchvision.models import resnet50, ResNet50_Weights
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        else:
            backbone = backbone_factory()
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layers = nn.ModuleList([backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4])
        self.lateral = nn.ModuleList(nn.Conv2d(c, channels, 1) for c in (256, 512, 1024, 2048))
        self.pm = nn.ModuleList(PerceptualModulation(2 * channels, channels, source_root, kernel_size,
                                                   dsconv_factory) for _ in range(3))
        self.semantic_adapter = cbr(channels, channels)
        self.scm = SpatialConstraint(channels)
        self.text_head, self.center_head = self._head(channels, 2), self._head(channels, 2)
        self.geometry_head = self._head(channels, 2)
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        self.provenance = dict(
            implementation='paper-guided reconstruction; not complete author implementation',
            paper='https://arxiv.org/html/2404.08965v3',
            backbone='torchvision ResNet50 IMAGENET1K_V2' if pretrained else 'random ResNet50 sanity only',
            channels=channels, output_stride=4, scm_stride=16,
            precision=dict(center_head='float32', text_head='float32', pm_conv_bn='float32'),
            attention=dict(axis='channel', activation='relu', scale='sqrt(spatial_locations)',
                           accumulation_dtype='float32', heads=1, learned_temperature=False, residual=False),
            assumptions=['three PM stages at stride16/8/4', 'two-axis DSC concat and 1x1 projection',
                         'post-attention 1x1 projection', 'SCM two-class one-hot mask input',
                         'SCM F1 is the first PM output at stride16',
                         'main semantic adapter stays on image-only path', 'positive height via softplus',
                         'tangent angle pi-periodic via bounded scalar'],
        )
        self.provenance['dsconv'] = getattr(self.pm[0].snake_h, 'provenance', {})

    @staticmethod
    def _head(channels, outputs):
        return nn.Sequential(cbr(channels, channels), nn.Conv2d(channels, outputs, 1))

    def forward(self, image, *, with_scm=False, mask=None):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError('Seeing input must be B,3,H,W RGB')
        if with_scm and mask is None:
            raise ValueError('Training SCM needs an explicit text mask')
        if with_scm and not self.training:
            raise ValueError('SCM is training-only; inference cannot consume a GT mask')
        x = self.stem((image - self.mean) / self.std)
        features = []
        for layer, lateral in zip(self.layers, self.lateral):
            x = layer(x)
            features.append(lateral(x))
        f = features[-1]
        semantic = None
        for index, level in enumerate((2, 1, 0)):
            f = F.interpolate(f, size=features[level].shape[-2:], mode='bilinear', align_corners=False)
            f = self.pm[index](torch.cat((features[level], f), 1))
            if index == 0:
                scm_input = f
                f = self.semantic_adapter(f)
                semantic = f
        geometry = self.geometry_head(f)
        # Both classification convolutions exceeded FP16 range before BN
        # normalization on real SynthText batches (steps28860/32962).
        with torch.autocast(image.device.type, enabled=False):
            text_logits = self.text_head(f.float())
            center_logits = self.center_head(f.float())
        out = dict(text=text_logits, center=center_logits,
                   height=F.softplus(geometry[:, :1]), angle=torch.tanh(geometry[:, 1:]) * (math.pi / 2),
                   feat=f, sem_main=semantic,
                   input_hw=tuple(image.shape[-2:]), feature_hw=tuple(f.shape[-2:]))
        if with_scm:
            out.update(self.scm(scm_input, mask))
        return out


def build_detector(config):
    cfg = dict(config) if isinstance(config, Mapping) else vars(config)
    if cfg.get('type', 'seeing') != 'seeing':
        raise ValueError('V5 detector factory supports only explicit Seeing reconstruction')
    return SeeingDetector(channels=int(cfg.get('channels', 256)),
                          source_root=cfg.get('source_root', '/opt/dscnet'),
                          kernel_size=int(cfg.get('kernel_size', 9)),
                          pretrained=bool(cfg.get('pretrained', True)))
