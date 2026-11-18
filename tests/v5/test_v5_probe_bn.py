import torch
from torch import nn

from tools.v5_probe_bn import with_bn_stats


def test_swap_changes_running_statistics_only():
    model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
    weights = {k:v.clone() for k,v in model.state_dict().items()}
    statistics = {k:v.clone() for k,v in weights.items()}
    statistics['1.running_mean'].fill_(3)
    statistics['1.running_var'].fill_(5)
    statistics['1.num_batches_tracked'].fill_(7)
    statistics['1.weight'].fill_(8)
    mixed, keys = with_bn_stats(model, weights, statistics)
    assert set(keys) == {'1.running_mean','1.running_var','1.num_batches_tracked'}
    assert torch.equal(mixed['1.running_mean'], statistics['1.running_mean'])
    assert torch.equal(mixed['1.running_var'], statistics['1.running_var'])
    assert torch.equal(mixed['1.num_batches_tracked'], statistics['1.num_batches_tracked'])
    assert torch.equal(mixed['1.weight'], weights['1.weight'])
    assert torch.equal(mixed['0.weight'], weights['0.weight'])
