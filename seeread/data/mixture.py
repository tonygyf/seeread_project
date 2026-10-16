"""Multi-dataset mixing for S3-v3 (replaces the single-jsonl loader in train.py).
cfg.data.train_anns: list of jsonl;  cfg.data.mix_weights: list of floats (relative share of SAMPLES drawn from each set).
Semantics per instance are decided inside SeeReadDataset (P0-14): no-transcript -> det only, ignore=true -> nothing.
Config example:
data:
  train_anns: [data/annotations/totaltext_train_main.jsonl, data/annotations/icdar2015_train.jsonl, data/annotations/ctw1500_train_det.jsonl]
  mix_weights: [1.0, 1.0, 1.0]
"""
from __future__ import annotations
import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from .datasets import SeeReadDataset


def build_mixture(root, anns, weights, num_samples=None, **ds_kw):
    assert len(anns) == len(weights) and len(anns) > 0
    sets = [SeeReadDataset(root, a, **ds_kw) for a in anns]
    ds = ConcatDataset(sets)
    wsum = float(sum(weights))
    per_item = []
    for s, w in zip(sets, weights):
        per_item += [(w / wsum) / max(len(s), 1)] * len(s)       # each dataset gets share w/wsum of draws
    sampler = WeightedRandomSampler(torch.tensor(per_item, dtype=torch.double), num_samples or len(ds), replacement=True)
    return ds, sampler, [len(s) for s in sets]
