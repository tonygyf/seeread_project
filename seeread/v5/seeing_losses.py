"""V5 two-class detector losses, including pi-periodic tangent regression."""
from __future__ import annotations
import torch
from torch.nn import functional as F


def _masked_mean(values, mask):
    values = values.float()
    mask = mask.to(dtype=torch.float32)
    return (values * mask).sum() / mask.sum().clamp_min(1)


def _valid_at_scale(valid, size):
    # A coarse SCM location is ignored when ANY covered fine pixel is ignored;
    # nearest-neighbor sampling can otherwise make small don't-care regions
    # disappear and reintroduce their supervision at the auxiliary resolution.
    ignored = (~valid.bool())[:, None].float()
    return 1 - F.adaptive_max_pool2d(ignored, size)[:, 0]


def detection_loss(pred, targets, *, scm_target=None):
    """Batched targets; all ignore pixels are excluded from every denominator.

    height is direct full height in feature pixels. angle is tangent radians,
    modulo pi. scm_mask is ALREADY two-channel sigmoid probability, not logits.
    Auxiliary semantics are detached, so SCM reconstruction trains that branch
    while semantic MSE supervises sem_main only.
    """
    # Cast BEFORE elementwise squares/subtractions, not after a half overflow.
    pred = {k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v
            for k, v in pred.items()}
    device = pred['text'].device
    gt = {k: torch.as_tensor(v, device=device) for k, v in targets.items()}
    if gt['text'].ndim == 2:
        gt = {k: v[None] for k, v in gt.items()}
    valid = gt['valid'].bool()
    center = valid & gt['center'].bool()
    losses = {}
    for key in ('text', 'center'):
        if pred[key].shape[1] != 2:
            raise ValueError(f'{key} must contain two class logits')
        ce = F.cross_entropy(pred[key], gt[key].long(), reduction='none')
        losses[key] = _masked_mean(ce, valid)
    height = pred['height'][:, 0]
    losses['height'] = _masked_mean(F.smooth_l1_loss(height, gt['height'].to(height.dtype), reduction='none'), center)
    delta = pred['angle'][:, 0] - gt['angle']
    wrapped = .5 * torch.atan2(torch.sin(2 * delta), torch.cos(2 * delta))
    losses['angle'] = _masked_mean(F.smooth_l1_loss(wrapped, torch.zeros_like(wrapped), reduction='none'), center)
    if 'scm_mask' in pred:
        scm = pred['scm_mask']
        if scm.shape[1] != 2:
            raise ValueError('scm_mask must contain two sigmoid probability channels')
        if scm_target is None:
            raise ValueError('SCM reconstruction requires the input-resolution text mask')
        size = scm.shape[-2:]
        source = torch.as_tensor(scm_target, device=device)
        if source.ndim == 2:
            source = source[None]
        if source.ndim != 3 or source.shape[0] != scm.shape[0]:
            raise ValueError('SCM reconstruction mask must have shape B,H,W')
        label = F.interpolate(source[:, None].float(), size=size, mode='nearest')[:, 0].long()
        weight = _valid_at_scale(valid, size)
        onehot = F.one_hot(label, 2).permute(0, 3, 1, 2).to(scm.dtype)
        losses['scm_mask'] = _masked_mean((scm - onehot).abs().mean(1), weight)
    if 'scm_sem' in pred:
        if 'sem_main' not in pred:
            raise ValueError('Semantic loss requires both scm_sem and sem_main')
        main, auxiliary = pred['sem_main'], pred['scm_sem']
        if main.shape != auxiliary.shape:
            raise ValueError('SCM and main semantic tensors must have identical shape')
        weight = _valid_at_scale(valid, main.shape[-2:])
        losses['scm_sem'] = _masked_mean((main - auxiliary.detach()).square().mean(1), weight)
    return losses
