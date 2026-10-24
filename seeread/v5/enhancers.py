"""Independent v5 enhancement baselines; no legacy detector context or IAE.

Self-DACE executes the author's *old_version* model.py unchanged. Stage 1
returns light_net(x)[0]; stage 2 additionally applies denoise_net. The official
test scripts consume RGB / 255 without further normalization, then save_image
clips the output. ``clamp_output=False`` exposes the raw upstream tensor.

The published Reading paper does not identify its precise Self-DACE checkpoint
or stage. These are explicit, public Self-DACE baselines, not its joint weights.
"""
from __future__ import annotations

import hashlib
import importlib.util
from collections.abc import Mapping
from pathlib import Path
import subprocess

import torch
from torch import nn


SELF_DACE_REPOSITORY = "https://github.com/John-Wendell/Self-DACE"
SELF_DACE_REVISION = "ec0c033def6c98a97b99c2938e7c8ffc50ba65fd"


def _get(config, key, default=None):
    return config.get(key, default) if isinstance(config, Mapping) else getattr(config, key, default)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file(root, configured, default):
    path = Path(configured).expanduser() if configured else root / default
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Self-DACE file is missing: {path}")
    return path


def _load_model_source(path):
    # Unique module name and no sys.path changes: both upstream stages are named
    # model.py, and ordinary `import model` can silently select the wrong one.
    name = "_seeread_v5_self_dace_" + _sha256(path)[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Self-DACE model source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_weights(module, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    # Official stage 1 is a direct state_dict; stage 2 wraps it under `net`.
    if isinstance(state, Mapping) and "net" in state:
        state = state["net"]
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"Invalid Self-DACE checkpoint: {path}")
    module.load_state_dict(state, strict=True)


def _validate_input(crops):
    if crops.ndim != 4 or crops.shape[1] != 3 or min(crops.shape) < 1:
        raise ValueError("Enhancer input must be a nonempty RGB tensor of shape N,3,H,W")
    if not crops.is_floating_point():
        raise TypeError("Enhancer input must use floating point RGB values in [0,1]")
    if not torch.isfinite(crops).all() or crops.detach().amin() < 0 or crops.detach().amax() > 1:
        raise ValueError("Enhancer input RGB values must be finite and in [0,1]")


class IdentityEnhancer(nn.Module):
    """Identity baseline with no optional source checkout or imports."""

    provenance = {"type": "identity"}

    def forward(self, crops):
        _validate_input(crops)
        return crops


class SelfDACEEnhancer(nn.Module):
    """Adapter around the unchanged official old-version Self-DACE modules.

    ``source_root`` is the external checkout root. Checkpoint paths can be
    absolute or relative to that root; omitted paths use upstream's shipped
    pre_train weights. Frozen parameters still allow derivatives w.r.t. input.
    """

    def __init__(self, source_root, stage=2, checkpoint=None,
                 denoise_checkpoint=None, freeze=True, clamp_output=True):
        super().__init__()
        if stage not in (1, 2, "1", "2"):
            raise ValueError("Self-DACE stage must be 1 (light) or 2 (light + denoise)")
        if not source_root:
            raise ValueError("Self-DACE requires an explicit source_root pointing to the official checkout")
        self.stage = int(stage)
        self.frozen = bool(freeze)
        self.clamp_output = bool(clamp_output)
        root = Path(source_root).expanduser().resolve()
        prefix = Path("codes_SelfDACE/old_version") / f"stage{self.stage}"
        source = _file(root, None, prefix / "model.py")
        upstream = _load_model_source(source)
        self.light = upstream.light_net()
        light_path = _file(root, checkpoint, prefix / "snapshots_light/pre_train_1.pth")
        _load_weights(self.light, light_path)
        self.denoise = None
        denoise_path = None
        if self.stage == 2:
            self.denoise = upstream.denoise_net()
            denoise_path = _file(root, denoise_checkpoint, prefix / "snapshots_denoise/pre_train_2.pth")
            _load_weights(self.denoise, denoise_path)
        elif denoise_checkpoint:
            raise ValueError("denoise_checkpoint is only valid for Self-DACE stage 2")
        try:
            revision = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=False,
            ).stdout.strip() or None
        except OSError:
            revision = None  # Archive installs need not include git; hashes still identify files.
        self.provenance = {
            "type": "self_dace", "repository": SELF_DACE_REPOSITORY,
            "source_root": str(root), "source_revision": revision,
            "source_file": str(source), "source_sha256": _sha256(source),
            "stage": self.stage, "checkpoint": str(light_path),
            "checkpoint_sha256": _sha256(light_path),
            "denoise_checkpoint": str(denoise_path) if denoise_path else None,
            "denoise_checkpoint_sha256": _sha256(denoise_path) if denoise_path else None,
            "clamp_output": self.clamp_output, "frozen": self.frozen,
            "weight_origin": "official Self-DACE public pretrain; not Reading joint weights",
        }
        self.requires_grad_(not self.frozen)
        if self.frozen:
            self.eval()

    def train(self, mode=True):
        # Preserve parameter and buffer freezing when the parent calls train().
        return super().train(False if self.frozen else mode)

    def forward(self, crops):
        _validate_input(crops)
        output = self.light(crops)[0]
        if self.denoise is not None:
            output = self.denoise(output)
        return output.clamp(0, 1) if self.clamp_output else output


def build_enhancer(config):
    """Build identity or genuine upstream Self-DACE, without implicit fallback."""
    kind = str(_get(config, "type", "identity")).lower()
    if kind == "identity":
        return IdentityEnhancer()
    if kind in ("srdace", "self_dace"):
        return SelfDACEEnhancer(
            source_root=_get(config, "source_root"), stage=_get(config, "stage", 2),
            checkpoint=_get(config, "checkpoint"),
            denoise_checkpoint=_get(config, "denoise_checkpoint"),
            freeze=_get(config, "freeze", True),
            clamp_output=_get(config, "clamp_output", True),
        )
    if kind in ("rllie", "iae"):
        raise NotImplementedError(
            f"{kind} has no verified official Reading adapter; the legacy IAE is not a substitute"
        )
    raise ValueError(f"Unknown v5 enhancer type: {kind}")
