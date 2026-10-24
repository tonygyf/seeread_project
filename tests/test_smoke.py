"""CPU smoke tests: python -m pytest -q tests  (or python tests/test_smoke.py). No downloads needed (CRNN recognizer, random-init backbone)."""
import os, sys, json, random, subprocess, tempfile
import numpy as np, cv2, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seeread.config import _ns, load_config
from seeread.data.lowlight_synth import darken, PHYS_LEVELS, illum_bin, luminance
from seeread.data.datasets import SeeReadDataset, collate
from seeread.models.seeread import SeeRead
from seeread.models.recognizer import CRNNRecognizer
from seeread.models.rectify import rectify_strip
from seeread.models.tsr import tsr_decode
from seeread.metrics.spotting import SpottingEvaluator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg(stage="joint", det=1.0):
    return _ns(dict(model=dict(det_ch=32, pretrained=False, ctx_dim=32, crop_hw=[32, 128], enh_iters=3, use_ctx=True, use_scm=True),
                    loss=dict(det=det, rec=1.0, enh=0.1, cic=0.5), train=dict(stage=stage)))


CFG = _cfg()


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


def test_p005_pred_heights_only_in_joint():
    import seeread.models.seeread as S
    with tempfile.TemporaryDirectory() as d:
        ann = _toy_dataset(d); ds = SeeReadDataset(d, ann, train=True, size=128, rec_k=5)
        batch = collate([ds[0], ds[0]])
        def n_sample_map_calls(stage):
            torch.manual_seed(0); random.seed(0)
            m = SeeRead(CRNNRecognizer(), _cfg(stage=stage))
            calls, orig = [], S.sample_map
            S.sample_map = lambda *a, **k: calls.append(1) or orig(*a, **k)
            try:
                m.forward_train(batch, pred_geom_prob=1.0)
            finally:
                S.sample_map = orig
            return len(calls)
        assert n_sample_map_calls("rec") == 0      # rec: fixed GT heights, curriculum never advances (P0-05)
        assert n_sample_map_calls("joint") > 0     # joint: predicted-height curriculum active


def test_p008_empty_encoding_skipped_not_class1():
    rec = CRNNRecognizer(); crops = torch.rand(2, 3, 32, 128)
    assert rec.cs.encode("停止") == []                        # pure non-charset text encodes to empty
    l = rec.forward_loss(crops, ["停止", "stop"])             # mixed batch: valid text trains, OOV skipped (no fake class 1)
    assert l is not None and l.requires_grad and rec.n_skip == 1
    assert rec.forward_loss(crops, ["停止", "中文"]) is None   # all-OOV batch: caller skips instead of "0"-labels (P0-08)
    assert rec.n_skip == 3


def test_p009_mixed_batch_cic_applies_to_paired_only():
    import seeread.models.seeread as S
    with tempfile.TemporaryDirectory() as d:
        ann = _toy_dataset(d)
        row = dict(image="images/a.png", instances=[dict(poly=[[40, 80], [180, 80], [180, 135], [40, 135]], text="stop")], meta=dict(illum_kind="real"))
        ann2 = os.path.join(d, "ann_nopair.jsonl"); open(ann2, "w").write(json.dumps(row))
        ds_p = SeeReadDataset(d, ann, train=True, size=128, rec_k=5)       # paired (lit kind)
        ds_u = SeeReadDataset(d, ann2, train=True, size=128, rec_k=5)      # unpaired
        batch = collate([ds_p[0], ds_u[0]])
        assert batch["lit_mask"].tolist() == [True, False]
        torch.manual_seed(0); random.seed(0)
        m = SeeRead(CRNNRecognizer(), CFG)
        got, orig = {}, S.cic_loss
        S.cic_loss = lambda f, g: (got.setdefault("n", int(f.shape[0])), orig(f, g))[1]
        try:
            L = m.forward_train(batch, pred_geom_prob=0.0)
        finally:
            S.cic_loss = orig
        assert "cic" in L and got["n"] == 1    # paired sample's single instance only; unpaired sample excluded (P0-09)


def test_p011_synth_dark_recomputes_instance_bin():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "images"))
        img = np.full((120, 160, 3), 230, np.uint8); cv2.putText(img, "STOP", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 3)
        cv2.imwrite(os.path.join(d, "images/a.png"), img[..., ::-1])
        row = dict(image="images/a.png", instances=[dict(poly=[[15, 45], [120, 45], [120, 90], [15, 90]], text="stop", bin="lit")], meta={})
        src = os.path.join(d, "src.jsonl"); open(src, "w").write(json.dumps(row))
        out = os.path.join(d, "out")
        subprocess.run([sys.executable, os.path.join(ROOT, "tools", "data", "make_synth_dark.py"), "--ann", src, "--root", d, "--out", out, "--seed", "1234"], check=True)
        rows = [json.loads(l) for l in open(os.path.join(out, "ann.jsonl"), encoding="utf-8")]
        assert len(rows) == len(PHYS_LEVELS)
        for r in rows:
            ins = r["instances"][0]
            assert ins["bin_source"] == "lit"                    # source bin preserved under a separate field (P0-11)
            dk = cv2.cvtColor(cv2.imread(os.path.join(d, r["image"])), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0   # image path is relative to root
            p = np.asarray(ins["poly"]); x0, y0 = np.maximum(p.min(0).astype(int), 0); x1, y1 = p.max(0).astype(int) + 1
            patch = dk[y0:y1, x0:x1]                             # instance bin re-derived from the DARK pixels
            assert ins["bin"] == illum_bin(luminance(patch))
        by_lv = {r["meta"]["level"]: r for r in rows}
        assert by_lv["extreme"]["instances"][0]["bin"] != "lit"


def test_p012_lit_image_untouched_at_eval():
    with tempfile.TemporaryDirectory() as d:
        ann = _toy_dataset(d)
        ds_tr = SeeReadDataset(d, ann, train=True, size=128, rec_k=5)
        ds_ev = SeeReadDataset(d, ann, train=False, size=128, rec_k=5)
        x_ev = ds_ev[0]
        orig = cv2.cvtColor(cv2.imread(os.path.join(d, "images/a.png")), cv2.COLOR_BGR2RGB)
        ref = cv2.resize(orig, (128, 96), interpolation=cv2.INTER_LINEAR)      # 240x320 -> s=0.4, canvas 128
        canvas = np.zeros((128, 128, 3), np.uint8); canvas[0:96, 0:128] = ref
        assert torch.equal(x_ev["image"], torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0)
        assert "lit" not in x_ev                                # eval: lit image loaded as-is, no online darkening (P0-12)
        x_tr = ds_tr[0]
        assert "lit" in x_tr and x_tr["image"].mean() < orig.astype(np.float32).mean() / 255.0   # train: darkened online


def test_p101_p104_rec_stage_skip_and_frozen_bn():
    from tools.legacy.train import set_frozen_eval
    m = SeeRead(CRNNRecognizer(), _cfg(stage="rec")); m.train(); set_frozen_eval(m, "rec")
    assert not m.det.training and m.enh.training and m.rec.training
    bn = next(mod for mod in m.det.modules() if isinstance(mod, torch.nn.BatchNorm2d))
    mu, sv = bn.running_mean.clone(), bn.running_var.clone()
    with tempfile.TemporaryDirectory() as d:
        ann = _toy_dataset(d); ds = SeeReadDataset(d, ann, train=True, size=128, rec_k=5)
        batch = collate([ds[0], ds[0]])
        m.forward_train(batch, pred_geom_prob=0.0)
        assert torch.equal(bn.running_mean, mu) and torch.equal(bn.running_var, sv)   # BN stats stay frozen (P1-04)
    with tempfile.TemporaryDirectory() as d:                    # rec stage, zero rec-eligible instances -> skip, not crash (P1-01)
        os.makedirs(os.path.join(d, "images"))
        cv2.imwrite(os.path.join(d, "images/a.png"), np.zeros((64, 64, 3), np.uint8))
        row = dict(image="images/a.png", instances=[dict(poly=[[10, 10], [40, 10], [40, 20], [10, 20]], text="###")], meta=dict(illum_kind="real"))
        ann = os.path.join(d, "ann.jsonl"); open(ann, "w").write("\n".join(json.dumps(row) for _ in range(4)))
        cfg_d = dict(model=dict(det_ch=32, pretrained=False, ctx_dim=32, crop_hw=[32, 128], enh_iters=3, use_ctx=True, use_scm=True),
                     data=dict(root=d, train_ann=ann, size=128, synth_mode="physical", workers=0), rec=dict(type="crnn"),
                     loss=dict(det=1.0, rec=1.0, enh=0.1, cic=0.5),
                     train=dict(stage="rec", epochs=1, batch_size=2, lr=1e-4, weight_decay=1e-4, pred_geom_warmup=0.5,
                                amp=False, seed=0, out_dir=os.path.join(d, "out"), resume=None, init_det=None, init_rec=None))
        cfg_p = os.path.join(d, "mini.yaml"); open(cfg_p, "w").write(yaml.safe_dump(cfg_d))
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "legacy", "train.py"), "--config", cfg_p],
                           capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), cwd=ROOT)
        assert r.returncode == 0, r.stderr[-2000:]
        saved = torch.load(os.path.join(d, "out", "last.pt"), map_location="cpu", weights_only=False)
        assert saved["state"]["empty_skips"] == 2
        assert saved["state"]["global_step"] == 0
        assert os.path.exists(os.path.join(d, "out", "last.pt"))


def test_p203_base_resolved_relative_to_config_dir():
    with tempfile.TemporaryDirectory() as d:
        sub = os.path.join(d, "configs"); os.makedirs(sub)
        open(os.path.join(sub, "base.yaml"), "w").write("loss: {det: 2.0, rec: 1.0}\n")
        open(os.path.join(sub, "exp.yaml"), "w").write("_base_: base.yaml\nloss: {rec: 3.0}\n")
        cwd = os.getcwd()
        try:
            os.chdir(tempfile.gettempdir())                     # different cwd, absolute config path
            c = load_config(os.path.join(sub, "exp.yaml"))
            assert c.loss.det == 2.0 and c.loss.rec == 3.0
            os.chdir(sub)                                       # bare config name, base next to it
            assert load_config("exp.yaml").loss.rec == 3.0
            b = os.path.join(sub, "base_abs.yaml"); open(b, "w").write("loss: {det: 5.0}\n")
            open(os.path.join(sub, "exp2.yaml"), "w").write(f"_base_: {b}\n")
            assert load_config(os.path.join(sub, "exp2.yaml")).loss.det == 5.0   # absolute _base_ supported (P2-03)
        finally:
            os.chdir(cwd)


def test_p206_scm_sr_ss_scaled_by_det_weight():
    with tempfile.TemporaryDirectory() as d:
        ann = _toy_dataset(d); ds = SeeReadDataset(d, ann, train=False, size=128, rec_k=5)
        batch = collate([ds[0]])
        def run(det_w):
            torch.manual_seed(0); random.seed(0)
            m = SeeRead(CRNNRecognizer(), _cfg(det=det_w))
            torch.manual_seed(1)
            return m.forward_train(batch, pred_geom_prob=0.0)
        a, b = run(1.0), run(2.0)
        assert set(a) == set(b)
        for k in a:                                             # sr/ss now scale with lambda_det like seg/h/dir (P2-06)
            r = b[k].item() / a[k].item()
            assert abs(r - (2.0 if k.startswith("det_") else 1.0)) < 1e-4, (k, r)


def test_p013_hash_text_still_learns_det_geometry():
    """LATeD-style textless ("###") instances must produce det targets (text/center/height) while
    staying out of rec crops; only out-of-canvas polys are det-ignored."""
    with tempfile.TemporaryDirectory() as d:
        ann = _toy_dataset(d); ds = SeeReadDataset(d, ann, train=False, size=128, rec_k=5)
        s = ds[0]
        assert int((s["targets"]["text"] > 0.5).sum()) > 0          # "###" poly IS a det positive
        assert int((s["targets"]["center"] > 0.5).sum()) > 0 and int((s["targets"]["height"] > 0).sum()) > 0
        assert len(s["instances"]) == 1                             # rec crop only for the real text
        assert s["instances"][0]["text"] == "stop"


def test_val_hook_saves_best_pt():
    """train.val_every>0: bright-val E2E F1 is evaluated periodically and best.pt is saved alongside last.pt."""
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "images"))
        img = np.full((240, 320, 3), 200, np.uint8); cv2.putText(img, "STOP", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (20, 20, 20), 4)
        cv2.imwrite(os.path.join(d, "images", "a.png"), img[..., ::-1])
        poly = [[40, 80], [180, 80], [180, 135], [40, 135]]
        row = json.dumps(dict(image="images/a.png", instances=[dict(poly=poly, text="stop")], meta=dict(illum_kind="lit")))
        ann = os.path.join(d, "train.jsonl"); open(ann, "w").write("\n".join([row] * 4))
        val = os.path.join(d, "val.jsonl"); open(val, "w").write(row)
        cfg_d = dict(model=dict(det_ch=32, pretrained=False, ctx_dim=32, crop_hw=[32, 128], enh_iters=3, use_ctx=True, use_scm=True),
                     data=dict(root=d, train_ann=ann, val_ann=val, size=128, synth_mode="physical", workers=0),
                     rec=dict(type="crnn"), loss=dict(det=1.0, rec=1.0, enh=0.1, cic=0.5),
                     train=dict(stage="joint", epochs=2, batch_size=2, lr=1e-4, weight_decay=1e-4, pred_geom_warmup=0.5,
                                amp=False, seed=0, out_dir=os.path.join(d, "out"), resume=None,
                                init_det=None, init_rec=None, val_every=1))
        cfg_p = os.path.join(d, "mini.yaml"); open(cfg_p, "w").write(yaml.safe_dump(cfg_d))
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "legacy", "train.py"), "--config", cfg_p],
                           capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), cwd=ROOT)
        assert r.returncode == 0, r.stderr[-2000:]
        assert r.stdout.count("e2e_F=") == 2
        assert os.path.exists(os.path.join(d, "out", "last.pt"))
        assert os.path.exists(os.path.join(d, "out", "best.pt"))                # peak-epoch checkpoint exists
        best = torch.load(os.path.join(d, "out", "best.pt"), map_location="cpu", weights_only=False)
        events = [json.loads(line) for line in open(os.path.join(d, "out", "train_events.jsonl"))]
        scores = [event["e2e_F"] for event in events
                  if event["event"] == "validation" and not event["partial"]]
        assert len(scores) == 2 and best["state"]["best_f1"] == max(scores)


def test_p014_eval_semantics_split():
    """ignore=true absorbs (no GT); '###'+ignore=false counts for detection but not e2e (LATeD/CTW)."""
    sq = [[0, 0], [10, 0], [10, 5], [0, 5]]
    ev = SpottingEvaluator()                                   # textless GT, correctly detected
    ev.add([dict(poly=sq, text="###", score=0.9)], [dict(poly=sq, text="###", bin="extreme")])
    a = ev.summary()["all"]
    assert a["n_gt"] == 1 and a["n_gt_e2e"] == 0 and abs(a["det_R"] - 1.0) < 1e-9 and abs(a["det_F"] - 1.0) < 1e-9
    assert abs(a["e2e_R"]) < 1e-9 and ev.summary()["extreme"]["det_recall"] == 1.0
    ev2 = SpottingEvaluator()                                  # explicit don't-care: absorbed
    ev2.add([dict(poly=sq, text="x", score=0.9)], [dict(poly=sq, text="###", ignore=True)])
    a2 = ev2.summary()["all"]
    assert a2["n_gt"] == 0 and a2["det_P"] == 0.0 and a2["det_R"] == 0.0
    ev3 = SpottingEvaluator()                                  # mixed: one transcribed + one textless
    tq = [[50, 50], [60, 50], [60, 55], [50, 55]]
    ev3.add([dict(poly=sq, text="stop", score=0.9), dict(poly=tq, text="z", score=0.8)],
            [dict(poly=sq, text="stop", bin="dark"), dict(poly=tq, text="###", bin="extreme")])
    a3 = ev3.summary()["all"]
    assert a3["n_gt"] == 2 and a3["n_gt_e2e"] == 1
    assert abs(a3["det_R"] - 1.0) < 1e-9 and abs(a3["e2e_R"] - 1.0) < 1e-9      # e2e denominator excludes ###


def test_p014_ignore_true_opt_out_of_det_training():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "images"))
        img = np.full((120, 160, 3), 230, np.uint8); cv2.putText(img, "STOP", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 3)
        cv2.imwrite(os.path.join(d, "images", "a.png"), img[..., ::-1])
        poly = [[15, 45], [120, 45], [120, 90], [15, 90]]
        row_ign = json.dumps(dict(image="images/a.png", instances=[dict(poly=poly, text="###", ignore=True)], meta={}))
        ann_i = os.path.join(d, "ign.jsonl"); open(ann_i, "w").write(row_ign)
        ds_i = SeeReadDataset(d, ann_i, train=False, size=128, rec_k=5)
        assert float(ds_i[0]["targets"]["text"].sum()) == 0.0      # protocol don't-care -> no det target
        row_txt = json.dumps(dict(image="images/a.png", instances=[dict(poly=poly, text="###", ignore=False)], meta={}))
        ann_t = os.path.join(d, "txt.jsonl"); open(ann_t, "w").write(row_txt)
        ds_t = SeeReadDataset(d, ann_t, train=False, size=128, rec_k=5)
        s = ds_t[0]
        assert float(s["targets"]["text"].sum()) > 0.0            # textless confirmed text -> learns det
        assert len(s["instances"]) == 0                            # but still no rec crop


def test_p015_estr_crop_contract():
    """ESTRCrops yields (dark, lit, label) with matched 3x64x256 tensors and drops un-transcriptable rows."""
    from tools.legacy.train_s25 import ESTRCrops, CROP_HW
    with tempfile.TemporaryDirectory() as d:
        for sub in ("low_light_train", "well_lit_train"):
            os.makedirs(os.path.join(d, sub))
            for i in range(3):
                pic = np.full((40, 120, 3), 60 if "low" in sub else 200, np.uint8)
                cv2.imwrite(os.path.join(d, sub, f"word_{i+1}.png"), pic)
        rows = []
        for i, t in enumerate(["stop", "go", "###"]):
            rows.append(json.dumps(dict(low_light=f"low_light_train/word_{i+1}.png",
                                        well_lit=f"well_lit_train/word_{i+1}.png", label=t)))
        open(os.path.join(d, "train.jsonl"), "w").write("\n".join(rows))
        ds = ESTRCrops("train", root=d)
        assert len(ds) == 2                                   # "###" (no alnum) dropped
        dark, lit, label = ds[0]
        assert tuple(dark.shape) == (3, CROP_HW[0], CROP_HW[1])
        assert dark.shape == lit.shape and label == "stop"


def test_p015_s25_mixed_replay_runs():
    """S2.5 mixes SynthText strips + ESTR real pairs in one optim step and saves a checkpoint (CPU, no init weights)."""
    with tempfile.TemporaryDirectory() as d:
        # --- synthetic full-image source (lit -> online darkening, with transcription) ---
        os.makedirs(os.path.join(d, "images"))
        img = np.full((256, 256, 3), 210, np.uint8); cv2.putText(img, "STOP", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, (20, 20, 20), 4)
        cv2.imwrite(os.path.join(d, "images", "a.png"), img[..., ::-1])
        poly = [[40, 90], [210, 90], [210, 170], [40, 170]]
        row = json.dumps(dict(image="images/a.png", instances=[dict(poly=poly, text="stop")], meta=dict(illum_kind="lit")))
        ann = os.path.join(d, "train.jsonl"); open(ann, "w").write("\n".join([row] * 4))
        # --- ESTR-style real paired crop source ---
        er = os.path.join(d, "estr")
        for sub in ("low_light_train", "well_lit_train"):
            os.makedirs(os.path.join(er, sub))
            for i in range(4):
                pic = np.full((40, 120, 3), 50 if "low" in sub else 200, np.uint8)
                cv2.putText(pic, "GO", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200) if "low" in sub else (20, 20, 20), 2)
                cv2.imwrite(os.path.join(er, sub, f"word_{i+1}.png"), pic)
        open(os.path.join(er, "train.jsonl"), "w").write("\n".join(
            json.dumps(dict(low_light=f"low_light_train/word_{i+1}.png",
                            well_lit=f"well_lit_train/word_{i+1}.png", label="go")) for i in range(4)))
        cfg_d = dict(model=dict(det_ch=32, pretrained=False, ctx_dim=32, crop_hw=[64, 256], enh_iters=3, use_ctx=True, use_scm=True),
                     data=dict(root=d, train_ann=ann, size=256, synth_mode="physical", workers=0),
                     rec=dict(type="crnn"), loss=dict(det=1.0, rec=1.0, enh=0.1, enh_rec=1.0, cic=0.5),
                     train=dict(stage="rec", epochs=1, batch_size=2, lr=1e-4, weight_decay=1e-4, pred_geom_warmup=0.5,
                                amp=False, seed=0, out_dir=os.path.join(d, "out"), resume=None,
                                init_det=None, init_rec=None, val_every=0))
        cfg_p = os.path.join(d, "mini.yaml"); open(cfg_p, "w").write(yaml.safe_dump(cfg_d))
        out = os.path.join(d, "s25")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "legacy", "train_s25.py"), "--config", cfg_p,
                            "--out", out, "--steps", "2", "--bs", "4", "--estr-frac", "0.5",
                            "--img-bs", "2", "--estr-workers", "0", "--estr-root", er],
                           capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), cwd=ROOT)
        assert r.returncode == 0, r.stderr[-2000:]
        assert "S25_DONE" in r.stdout and os.path.exists(os.path.join(out, "last.pt"))


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("PASS", k)
