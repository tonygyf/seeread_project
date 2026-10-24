"""S3 regressions. Run with python tests/legacy/test_s3_regressions.py [--device cuda]."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from seeread.config import _ns
from seeread.data.datasets import SeeReadDataset
from seeread.models.recognizer import CRNNRecognizer
from seeread.models.seeread import SeeRead
from tools.legacy.train import set_frozen_eval


def test_frozen_rec_keeps_state_and_input_gradients(device="cpu"):
    torch.manual_seed(0)
    cfg = _ns(dict(model=dict(det_ch=32, pretrained=False, ctx_dim=32, crop_hw=[32, 128],
                             enh_iters=3, use_ctx=True), train=dict(stage="joint", freeze_rec=True)))
    model = SeeRead(CRNNRecognizer(), cfg).to(device)
    model.rec.requires_grad_(False)
    before = {k: v.detach().clone() for k, v in model.rec.state_dict().items()}
    # Both training-loop entry points call model.train(), including after validation.
    for _ in range(2):
        model.train()
        set_frozen_eval(model, "joint")
        x = torch.rand(2, 3, 32, 128, device=device, requires_grad=True)
        y, _ = model.enh(x, torch.zeros(2, 32, device=device))
        loss = model.rec.forward_loss(y, ["stop", "go"])
        with torch.no_grad():
            model.rec.encode_features(x)
        features = model.rec.encode_features(y)
        loss = loss + features.square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        assert model.enh.head.weight.grad is not None and model.enh.head.weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in model.rec.parameters())
        assert all(torch.equal(before[k], v) for k, v in model.rec.state_dict().items())


def test_partial_word_excluded_from_rec():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cv2.imwrite(str(root / "image.png"), np.full((128, 128, 3), 180, np.uint8))
        inside = dict(poly=[[20, 40], [90, 40], [90, 70], [20, 70]], text="stop")
        clipped = dict(poly=[[-20, 80], [90, 80], [90, 110], [-20, 110]], text="partial")
        ann = root / "ann.jsonl"
        ann.write_text(json.dumps(dict(image="image.png", instances=[inside, clipped])))
        sample = SeeReadDataset(tmp, str(ann), train=False, size=128)[0]
        assert [i["text"] for i in sample["instances"]] == ["stop"]
        assert sample["targets"]["text"].sum() > 0
        assert (sample["targets"]["weight"] == 0).any()


def test_darktext_checkpoint_override():
    from tools.legacy import eval_darktext_det as ev
    loaded = []

    class FakeTextNet:
        def __init__(self, **kwargs): pass
        def to(self, dev): return self
        def eval(self): pass
        def load_model(self, path):
            loaded.append(path)
            return [], []

    with tempfile.TemporaryDirectory() as tmp, \
         patch.object(ev, "TextNet", FakeTextNet), \
         patch.object(ev, "run_pass", return_value=[]), \
         patch.object(ev, "detection_metrics", return_value={}), \
         patch.object(sys, "argv", ["eval_darktext_det.py", "--ckpt", "/custom/weights.pth", "--dev", "cpu", "--out", tmp]):
        ev.main()
        assert loaded == ["/custom/weights.pth"]
        assert json.loads((Path(tmp) / "det_f_size640.json").read_text())["config"]["ckpt"] == loaded[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(4)
    test_frozen_rec_keeps_state_and_input_gradients(args.device)
    print("PASS frozen rec state and gradients", args.device)
    test_partial_word_excluded_from_rec()
    print("PASS partial word labels")
    test_darktext_checkpoint_override()
    print("PASS checkpoint override")
