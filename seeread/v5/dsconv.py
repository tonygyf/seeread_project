"""Pinned official 2D DSConv-pro operator, independent of the legacy Lite op.

Upstream: https://github.com/YaoleiQi/DSCNet (MIT), revision below.
This adapter imports the pinned source from an explicit external checkout.
It is an operator adapter, not an implementation of the complete Seeing paper.

Two compatibility details are explicit: coordinate devices follow each input,
and if_offset=False uses zero offsets (upstream pro stores but ignores its flag).
The historic non-pro implementation is intentionally unsupported: its endpoint
loop and batch indexing differ from the author's newer grid_sample version.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


DSCNET_REPOSITORY = "https://github.com/YaoleiQi/DSCNet"
DSCNET_REVISION = "0e35b2b6ea748a5e8ce7557ea3e580b9951471d6"
DSCONV_SOURCE_PATH = "DSCNet_2D_opensource/Code/DRIVE/S3_DSConv_pro.py"
DSCONV_SOURCE_SHA256 = "807ba97eab47d18737f3c7a565053ac0ed0365efb6b5aab931e2ab657e44e0b4"


def _load_source(root):
    path = root / DSCONV_SOURCE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Official DSConv-pro source is missing: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != DSCONV_SOURCE_SHA256:
        raise ValueError(
            f"DSConv source differs from pinned revision {DSCNET_REVISION}: {path}; "
            f"expected {DSCONV_SOURCE_SHA256}, found {digest}"
        )
    spec = importlib.util.spec_from_file_location("_seeread_v5_official_dsconv_pro", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import DSConv source: {path}")
    source = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(source)
    except ModuleNotFoundError as error:
        if error.name == "einops":
            raise ImportError("Official DSConv-pro requires einops; install it in the detector environment") from error
        raise
    return source, path, digest


class OfficialDSConv(nn.Module):
    """B,Cin,H,W -> B,Cout,H,W, initialized on CPU then movable with .to().

    source_root is the pinned DSCNet checkout. Upstream currently requires a
    Cout compatible with GroupNorm(Cout//4, Cout); K must be odd. No public
    medical-segmentation weights are loaded into text detection experiments.
    """

    def __init__(self, cin, cout, kernel_size=9, morph=0, extend_scope=1.0,
                 if_offset=True, source_root=None):
        super().__init__()
        if not source_root:
            raise ValueError("OfficialDSConv requires an explicit source_root for the pinned DSCNet checkout")
        if cin < 1 or cout < 4 or cout % (cout // 4):
            raise ValueError("DSConv needs cin >= 1 and cout compatible with GroupNorm(cout//4, cout)")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("DSConv kernel_size must be odd and at least 3")
        if morph not in (0, 1):
            raise ValueError("DSConv morph must be 0 or 1")
        if extend_scope < 0:
            raise ValueError("DSConv extend_scope must be nonnegative")
        root = Path(source_root).expanduser().resolve()
        self._source, path, digest = _load_source(root)
        self.cin, self.cout = cin, cout
        self.if_offset = bool(if_offset)
        self.operator = self._source.DSConv_pro(
            in_channels=cin, out_channels=cout, kernel_size=kernel_size,
            morph=morph, extend_scope=extend_scope, if_offset=if_offset,
            device="cpu",
        )
        try:
            revision = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=False,
            ).stdout.strip() or None
        except OSError:
            revision = None
        self.provenance = dict(
            repository=DSCNET_REPOSITORY, pinned_revision=DSCNET_REVISION,
            checkout_revision=revision, source_root=str(root), source_file=str(path),
            source_sha256=digest, variant="official_2d_pro", if_offset=self.if_offset,
            kernel_size=kernel_size, morph=morph, extend_scope=extend_scope,
            pretrained_weights=None,
            precision='deformed convolution and following GroupNorm in float32; normalized output returns to AMP dtype',
            activation_memory='recompute DSConv under training autocast; no random/stateful layers inside operator',
            compatibility=["coordinate device follows input", "disabled offsets explicitly zeroed"],
        )

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.cin or x.shape[0] < 1 or min(x.shape[-2:]) < 2:
            raise ValueError(f"DSConv expects nonempty B,{self.cin},H,W with H,W >= 2")
        if not x.is_floating_point():
            raise TypeError("DSConv expects floating point features")
        if self.training and torch.is_grad_enabled() and torch.is_autocast_enabled(x.device.type):
            # Expanded deformed maps are large (K times H/W). Recompute these
            # pure GroupNorm/geometry operations rather than retaining FP32
            # convolution inputs beyond the agreed single-GPU memory cap.
            return checkpoint(self._forward, x, use_reentrant=False, preserve_rng_state=False)
        return self._forward(x)

    def _forward(self, x):
        # Upstream stores a plain device attribute; Module.to() cannot update it.
        self.operator.device = x.device
        if self.if_offset:
            offsets = self.operator.tanh(self.operator.gn_offset(self.operator.offset_conv(x)))
        else:
            # Upstream forward does not honor if_offset.
            offsets = x.new_zeros((x.shape[0], 2 * self.operator.kernel_size, *x.shape[-2:]))
        y_map, x_map = self._source.get_coordinate_map_2D(
            offsets, morph=self.operator.morph,
            extend_scope=self.operator.extend_scope, device=x.device,
        )
        deformed = self._source.get_interpolated_feature(x, y_map, x_map)
        conv = self.operator.dsc_conv_x if self.operator.morph == 0 else self.operator.dsc_conv_y
        # Real step153502: finite deformed input, FP16 conv >65504 before GN.
        # Keep offsets/grid sampling in their original precision and preserve
        # the upstream layers/parameter paths. Full-operator FP32 exceeds the
        # agreed training memory cap; only this unnormalized branch needs it.
        dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        with torch.autocast(x.device.type, enabled=False):
            normalized = self.operator.relu(self.operator.gn(conv(deformed.float())))
        return normalized.to(dtype)
