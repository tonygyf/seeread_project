"""Teacher-state and deterministic dark-validation contracts."""
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from seeread.data.datasets import SeeReadDataset
from seeread.models.recognizer import BaseRecognizer


def test_teacher_is_stable_without_changing_student_modes_or_buffers():
    class TinyRecognizer(BaseRecognizer):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(3)
            self.dropout = nn.Dropout(0.9)
        def encode_features(self, x):
            return self.dropout(self.bn(x))
    rec = TinyRecognizer().train()
    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    buffers = {k: v.clone() for k, v in rec.named_buffers()}
    first = rec.teacher_features(x)
    second = rec.teacher_features(x)
    assert torch.equal(first, second) and not first.requires_grad
    assert rec.training and rec.bn.training and rec.dropout.training
    assert all(torch.equal(v, buffers[k]) for k, v in rec.named_buffers())
    rec.bn.eval()
    rec.teacher_features(x)
    assert rec.training and not rec.bn.training and rec.dropout.training
    rec.encode_features(x).mean().backward()
    assert x.grad is not None


def test_dark_validation_is_fixed_and_preserves_default_original():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        image = np.full((96, 128, 3), 220, np.uint8)
        cv2.imwrite(str(root / "a.png"), image)
        row = dict(image="a.png", instances=[dict(poly=[[10, 20], [90, 20], [90, 60], [10, 60]],
                                                   text="stop", bin="lit")], meta=dict(illum_kind="lit"))
        ann = root / "ann.jsonl"
        ann.write_text(json.dumps(row))
        original = SeeReadDataset(tmp, str(ann), train=False, size=128)[0]
        dark_ds = SeeReadDataset(tmp, str(ann), train=False, size=128, seed=1234, eval_condition="dark")
        dark, repeat = dark_ds[0], dark_ds[0]
        assert torch.equal(dark["image"], repeat["image"])
        assert float(dark["image"].mean()) < float(original["image"].mean())
        assert dark["scale"] == original["scale"] and dark["offset"] == original["offset"]
        assert dark["meta"]["eval_condition"] == "fixed_synthetic_dark"
        assert dark["gt_orig"][0]["bin"] != "lit"
        assert dark_ds.items[0]["instances"][0]["bin"] == "lit"
        assert original["gt_orig"][0]["bin"] == "lit"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
