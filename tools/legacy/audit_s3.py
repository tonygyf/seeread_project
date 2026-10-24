"""Bounded, read-only checkpoint audit on fixed validation GT crops.

Run inside the project container: python tools/legacy/audit_s3.py --max-images 24.
This is an attribution probe, not an E2E benchmark or a training procedure.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from seeread.config import load_config
from seeread.data.datasets import SeeReadDataset
from seeread.data.lowlight_synth import darken
from seeread.metrics.spotting import SpottingEvaluator, edit_distance, norm_text
from seeread.models.recognizer import build_recognizer
from seeread.models.rectify import rectify_strip, sample_map
from seeread.models.seeread import SeeRead
from seeread.models.tsr import tsr_decode


CHECKPOINTS = {
    "s2": "results/archive/legacy/v2/s2_rec_v2/last.pt",
    "s3_v2_best": "results/archive/legacy/v2/s3_joint_v2/best.pt",
    "s3_v3_ft_best": "results/archive/legacy/v3/s3_joint_v3_ft/best.pt",
    "s3_v3_ft_last": "results/archive/legacy/v3/s3_joint_v3_ft/last.pt",
    "s3_v3_frozen": "results/archive/legacy/v3/s3_joint_v3_frz/best.pt",
}


def metrics(preds, texts):
    pairs = [(norm_text(p), norm_text(t)) for p, t in zip(preds, texts)]
    return dict(n=len(pairs), correct=sum(p == t for p, t in pairs),
                word_acc=sum(p == t for p, t in pairs) / max(len(pairs), 1),
                cer=sum(edit_distance(p, t) for p, t in pairs) /
                    max(sum(len(t) for _, t in pairs), 1),
                blank_fraction=sum(not p for p, _ in pairs) / max(len(pairs), 1))


@torch.no_grad()
def factorial(model, states, ds, limit, dev):
    """Hold predicted polygons/crops fixed within each detector row; swap recognizers.

    Context comes from the row detector, as in deployment. Native-enhancer cells
    therefore test an enhancer+recognizer bundle, not the recognizer alone.
    """
    names = ("s3_v3_ft_best", "s3_v3_ft_last")
    result = {}
    for det_name in names:
        model.det.load_state_dict(states[det_name]["det"])
        evaluators = {(rec_name, enh): SpottingEvaluator()
                      for rec_name in names for enh in ("native", "identity")}
        for i in range(min(limit, len(ds))):
            sample = ds[i]
            x = sample["image"][None].to(dev)
            out = model.det(x)
            ctx = model.det.illum_context(x, out["c5"])
            geo = torch.cat([out["geo"][:, :1].clamp(min=0), out["geo"][:, 1:]], 1)[0].cpu().numpy()
            dets = tsr_decode(out["text"][0, 0].sigmoid().cpu().numpy(),
                              out["center"][0, 0].sigmoid().cpu().numpy(), geo, model.stride)[:100]
            crops = []
            for d in dets:
                c, h = d["centers"], d["heights"]
                if len(c) < 2:
                    c = np.concatenate([c - [2, 0], c + [2, 0]])
                    h = np.repeat(h, 2)
                crops.append(rectify_strip(x, torch.as_tensor(c, dtype=torch.float32, device=dev),
                                           torch.as_tensor(h, dtype=torch.float32, device=dev), model.crop_hw))
            for rec_name in names:
                model.rec.load_state_dict(states[rec_name]["rec"])
                model.enh.load_state_dict(states[rec_name]["enh"])
                for enh in ("native", "identity"):
                    preds = []
                    if crops:
                        batch = torch.cat(crops)
                        if enh == "native":
                            batch, _ = model.enh(batch, ctx.expand(len(dets), -1))
                        texts = model.rec.predict(batch)
                        offset = np.asarray(sample["offset"])
                        preds = [dict(poly=(d["poly"] - offset) / sample["scale"], score=d["score"], text=t)
                                 for d, t in zip(dets, texts)]
                    evaluators[(rec_name, enh)].add(preds, sample["gt_orig"],
                                                   image_bin=sample["meta"].get("illum_bin"))
        result[det_name] = {f"{rec_name}/{enh}": ev.summary()["all"]
                            for (rec_name, enh), ev in evaluators.items()}
        print("factorial", det_name, json.dumps(result[det_name]), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ann", default="data/annotations/totaltext_val.jsonl")
    ap.add_argument("--max-images", type=int, default=24)
    ap.add_argument("--out", default="results/archive/diagnostics/s3_audit/probe.json")
    ap.add_argument("--factorial", action="store_true", help="run lit validation E2E detector/recognizer swaps instead")
    args = ap.parse_args()
    if args.max_images < 1:
        ap.error("--max-images must be positive")
    torch.manual_seed(0)
    torch.set_num_threads(4)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config("configs/base.yaml", {"model": {"pretrained": False}})
    ds = SeeReadDataset(cfg.data.root, args.ann, train=False, size=cfg.data.size)
    states = {k: torch.load(p, map_location="cpu", weights_only=False) for k, p in CHECKPOINTS.items()}
    model = SeeRead(build_recognizer(cfg.rec), cfg).to(dev).eval()
    if args.factorial:
        report = dict(protocol="lit E2E 2x2 detector and enhancer/rec bundle swap, plus identity enhancer; "
                      "same crops within a detector row; current TSR implementation, not archived benchmark code",
                      ann=args.ann, n_images=min(args.max_images, len(ds)), checkpoints=CHECKPOINTS,
                      results=factorial(model, states, ds, args.max_images, dev))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        return
    samples = []
    # Dataset supplies exactly the same GT geometry and eligibility for both conditions.
    for i in range(min(args.max_images, len(ds))):
        sample = ds[i]
        raw = cv2.cvtColor(cv2.imread(os.path.join(ds.root, ds.items[i]["image"])), cv2.COLOR_BGR2RGB)
        dark, _ = darken(raw, np.random.default_rng((1234, i)), mode=cfg.data.synth_mode)
        h, w = raw.shape[:2]
        dark = ds._apply(dark, ds._params(h, w, np.random.default_rng(0)))
        sample["dark"] = torch.from_numpy(dark).permute(2, 0, 1).float() / 255
        samples.append(sample)
    s2rec = build_recognizer(cfg.rec).to(dev).eval()
    s2rec.load_state_dict(states["s2"]["rec"])
    report = dict(protocol="fixed GT centers, 16 eligible instances/image maximum; no detection matching; "
                  "same GT crops across checkpoints; darken before eval resize; diagnostic interventions only",
                  ann=args.ann, ann_sha256=hashlib.sha256(Path(args.ann).read_bytes()).hexdigest(),
                  n_images=len(samples), dark_seed=1234, crop_hw=list(cfg.model.crop_hw),
                  torch=torch.__version__, device=dev, checkpoints={}, results={})
    for name, state in states.items():
        model.det.load_state_dict(state["det"])
        model.enh.load_state_dict(state["enh"])
        model.rec.load_state_dict(state["rec"])
        model.eval()
        restored = copy.deepcopy(model.rec).eval()
        with torch.no_grad():
            for key, value in restored.named_buffers():
                value.copy_(states["s2"]["rec"][key].to(dev))
        report["checkpoints"][name] = dict(path=CHECKPOINTS[name], train_config=state.get("cfg"),
            recognizer="CRNN" if "fc.weight" in state["rec"] else "other",
            rec_parameter_max_delta_vs_s2=max(float((p.cpu() - states["s2"]["rec"][k]).abs().max())
                                             for k, p in model.rec.named_parameters()),
            bn_counts={k: int(v) for k, v in state["rec"].items() if k.endswith("num_batches_tracked")})
        report["results"][name] = {}
        for cond in ("lit", "dark"):
            predictions = {k: [] for k in ("native_gt", "identity_gt", "s2_bn_gt", "s2_rec_gt", "native_pred_height")}
            texts, height_ratios = [], []
            for sample in samples:
                instances = [ins for ins in sample["instances"] if norm_text(ins["text"])]
                if not instances:
                    continue
                x = sample["image" if cond == "lit" else "dark"][None].to(dev)
                with torch.no_grad():
                    out = model.det(x)
                    ctx = model.det.illum_context(x, out["c5"])
                    gt_crops, ph_crops = [], []
                    for ins in instances:
                        c = torch.from_numpy(ins["centers"]).to(dev)
                        h = torch.from_numpy(ins["heights"]).to(dev)
                        ph = torch.expm1(sample_map(out["geo"][:, :1], c, model.stride)[:, 0].clamp(min=0)) * model.stride
                        height_ratios.extend((ph / h.clamp(min=1e-6)).cpu().tolist())
                        gt_crops.append(rectify_strip(x, c, h, model.crop_hw))
                        ph_crops.append(rectify_strip(x, c, ph, model.crop_hw))
                    crops = torch.cat(gt_crops)
                    enhanced, _ = model.enh(crops, ctx.expand(len(instances), -1))
                    enhanced_ph, _ = model.enh(torch.cat(ph_crops), ctx.expand(len(instances), -1))
                    predictions["native_gt"].extend(model.rec.predict(enhanced))
                    predictions["identity_gt"].extend(model.rec.predict(crops))
                    predictions["s2_bn_gt"].extend(restored.predict(enhanced))
                    predictions["s2_rec_gt"].extend(s2rec.predict(enhanced))
                    predictions["native_pred_height"].extend(model.rec.predict(enhanced_ph))
                texts.extend(ins["text"] for ins in instances)
            result = {k: metrics(p, texts) for k, p in predictions.items()}
            result["pred_height_over_gt"] = dict(zip(("p05", "p50", "p95"),
                                                       np.percentile(height_ratios, [5, 50, 95]).tolist())) if height_ratios else None
            report["results"][name][cond] = result
            print(name, cond, json.dumps(result), flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print("Saved", args.out, flush=True)


if __name__ == "__main__":
    main()
