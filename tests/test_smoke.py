"""CPU smoke tests: python -m pytest -q tests  (or python tests/test_smoke.py). No downloads needed (CRNN recognizer, random-init backbone)."""
import os, sys, json, tempfile
import numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seeread.config import _ns
from seeread.data.lowlight_synth import darken, PHYS_LEVELS
from seeread.data.datasets import SeeReadDataset, collate
from seeread.models.seeread import SeeRead
from seeread.models.recognizer import CRNNRecognizer
from seeread.models.rectify import rectify_strip
from seeread.models.tsr import tsr_decode
from seeread.metrics.spotting import SpottingEvaluator

CFG = _ns(dict(model=dict(det_ch=32, pretrained=False, ctx_dim=32, crop_hw=[32, 128], enh_iters=3, use_ctx=True, use_scm=True),
               loss=dict(det=1.0, rec=1.0, enh=0.1, cic=0.5), train=dict(stage="joint")))


def _toy_dataset(d):
    os.makedirs(os.path.join(d, "images"), exist_ok=True)
    img = np.full((240, 320, 3), 200, np.uint8); cv2.putText(img, "STOP", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (20, 20, 20), 4)
    cv2.imwrite(os.path.join(d, "images/a.png"), img[..., ::-1])
    poly = [[40, 80], [180, 80], [180, 135], [40, 135]]               # top L->R, bottom R->L  (tl,tr,br,bl)
    rows = [dict(image="images/a.png", instances=[dict(poly=poly, text="stop"), dict(poly=[[200, 10], [260, 10], [260, 40], [200, 40]], text="###")],
                 meta=dict(illum_kind="lit"))]
    open(os.path.join(d, "ann.jsonl"), "w").write("\n".join(json.dumps(r) for r in rows))
    return os.path.join(d, "ann.jsonl")


def test_synth_levels_are_ordered():
    img = np.random.default_rng(0).integers(60, 255, (96, 128, 3), dtype=np.uint8)
    ys = [np.mean([darken(img, np.random.default_rng(s), lv)[1]["illum_y"] for s in range(5)]) for lv in PHYS_LEVELS]
    assert ys[0] > ys[1] > ys[2], ys


def test_rectify_is_differentiable_wrt_height():
    img = torch.rand(1, 3, 100, 200); h = torch.full((5,), 20.0, requires_grad=True)
    c = torch.stack([torch.linspace(30, 170, 5), torch.full((5,), 50.0)], 1)
    out = rectify_strip(img, c, h, (32, 128)); out.mean().backward()
    assert out.shape == (1, 3, 32, 128) and h.grad is not None and h.grad.abs().sum() > 0


def test_tsr_decodes_a_synthetic_bar():
    H = W = 64; t = np.zeros((H, W), np.float32); c = np.zeros_like(t); t[20:36, 8:56] = 1; c[26:30, 10:54] = 1
    geo = np.zeros((3, H, W), np.float32); geo[0] = np.log1p(14); geo[1] = 1
    res = tsr_decode(t, c, geo, stride=4)
    assert len(res) == 1 and res[0]["poly"].shape[1] == 2


def test_train_and_infer_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        ann = _toy_dataset(d); ds = SeeReadDataset(d, ann, train=True, size=128, rec_k=5)
        batch = collate([ds[0], ds[0]]); assert "lit" in batch
        m = SeeRead(CRNNRecognizer(), CFG); L = m.forward_train(batch, pred_geom_prob=1.0)
        assert {"det_seg_text", "det_sr", "det_ss", "rec", "enh", "cic"} <= set(L), set(L)
        sum(L.values()).backward()
        assert m.det.ctx_mlp[0].weight.grad is not None          # recognition path reaches detector ctx MLP
        res = m.infer(batch["image"][:1]); assert isinstance(res, list) and len(res) == 1


def test_evaluator():
    ev = SpottingEvaluator(); sq = [[0, 0], [10, 0], [10, 5], [0, 5]]
    ev.add([dict(poly=sq, text="Stop!", score=0.9)], [dict(poly=sq, text="stop", bin="dark"), dict(poly=[[50, 50], [60, 50], [60, 55], [50, 55]], text="x", bin="extreme")])
    s = ev.summary(); assert abs(s["all"]["e2e_R"] - 0.5) < 1e-6 and s["dark"]["e2e_recall"] == 1.0 and s["extreme"]["frac_missed"] == 1.0


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("PASS", k)
