"""Equal-step fixed-detector pilot: joint enhancer+CRNN vs identity+CRNN.

This deliberately freezes the detector and crop interface. It does not reproduce
the original all-module S3 training or establish a converged method ranking.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from seeread.config import load_config
from seeread.data.datasets import SeeReadDataset
from seeread.data.lowlight_synth import darken
from seeread.losses import cic_loss
from seeread.metrics.spotting import SpottingEvaluator, _poly, iou, norm_text
from seeread.models.enhancer import enhancer_regulariser
from seeread.models.recognizer import build_recognizer
from seeread.models.rectify import rectify_strip
from seeread.models.seeread import SeeRead
from seeread.models.tsr import tsr_decode

DET = "results/archive/legacy/v2/s3_joint_v2/best.pt"
INIT = "results/archive/legacy/v2/s2_rec_v2/last.pt"


def crops(model, image, geometries, dev):
    if not geometries:
        return torch.empty(0, 3, *model.crop_hw)
    x = image[None].to(dev)
    return torch.cat([rectify_strip(x, torch.as_tensor(g["centers"], dtype=torch.float32, device=dev),
                                   torch.as_tensor(g["heights"], dtype=torch.float32, device=dev), model.crop_hw)
                      for g in geometries]).cpu()


@torch.no_grad()
def cache(model, ds, limit, dev, seed, training):
    items, records = [], []
    for i in range(min(limit, len(ds))):
        sample = ds[i]
        raw = cv2.cvtColor(cv2.imread(os.path.join(ds.root, ds.items[i]["image"])), cv2.COLOR_BGR2RGB)
        dark, _ = darken(raw, np.random.default_rng((seed, i)), mode=ds.synth_mode)
        h, w = raw.shape[:2]
        dark = ds._apply(dark, ds._params(h, w, np.random.default_rng(0)))
        dark = torch.from_numpy(dark).permute(2, 0, 1).float() / 255
        for cond, image in (("lit", sample["image"]), ("dark", dark)):
            x = image[None].to(dev)
            out = model.det(x)
            ctx = model.det.illum_context(x, out["c5"])[0].cpu()
            geo = torch.cat([out["geo"][:, :1].clamp(min=0), out["geo"][:, 1:]], 1)[0].cpu().numpy()
            dets = tsr_decode(out["text"][0, 0].sigmoid().cpu().numpy(),
                              out["center"][0, 0].sigmoid().cpu().numpy(), geo, model.stride)[:100]
            for d in dets:
                if len(d["centers"]) < 2:
                    d["centers"] = np.concatenate([d["centers"] - [2, 0], d["centers"] + [2, 0]])
                    d["heights"] = np.repeat(d["heights"], 2)
            pred_crops = crops(model, image, dets, dev)
            gt_geos = [g for g in sample["instances"] if norm_text(g["text"])]
            gt_crops = crops(model, image, gt_geos, dev)
            if not training:
                records.append(dict(cond=cond, pred_crops=pred_crops.half(), gt_crops=gt_crops.half(),
                                    ctx=ctx, dets=dets, scale=sample["scale"], offset=sample["offset"],
                                    gts=sample["gt_orig"], texts=[g["text"] for g in gt_geos]))
                continue
            # Same supervised examples for both arms; paired lit target uses identical geometry.
            teacher_gt = crops(model, sample["image"], gt_geos, dev)
            for j, g in enumerate(gt_geos):
                items.append((gt_crops[j].half(), teacher_gt[j].half(), ctx, g["text"], cond, "gt"))
            used = set()
            gts = sample["gt_orig"]
            gp = [_poly(g["poly"]) for g in gts]
            for j in sorted(range(len(dets)), key=lambda j: -dets[j]["score"]):
                d = dets[j]
                poly = _poly((d["poly"] - np.asarray(sample["offset"])) / sample["scale"])
                candidates = [(iou(poly, p), k) for k, p in enumerate(gp) if k not in used]
                score, k = max(candidates, default=(0.0, -1))
                if score < 0.5:
                    continue
                used.add(k)
                if gts[k].get("ignore") or not norm_text(gts[k].get("text", "")):
                    continue
                teacher = crops(model, sample["image"], [d], dev)[0]
                items.append((pred_crops[j].half(), teacher.half(), ctx, gts[k]["text"], cond, "pred"))
        if (i + 1) % 20 == 0:
            print("cache", "train" if training else "val", i + 1, flush=True)
    return items if training else records


@torch.no_grad()
def evaluate(rec, enh, records, dev, bs):
    rec.eval()
    if enh is not None:
        enh.eval()
    ev = {c: SpottingEvaluator() for c in ("lit", "dark")}
    gt_counts = {c: [0, 0] for c in ev}
    for r in records:
        def predict(xs):
            output = []
            for j in range(0, len(xs), bs):
                x = xs[j:j + bs].float().to(dev)
                if enh is not None:
                    x, _ = enh(x, r["ctx"].to(dev)[None].expand(len(x), -1))
                output.extend(rec.predict(x))
            return output
        texts = predict(r["pred_crops"])
        preds = [dict(poly=(d["poly"] - np.asarray(r["offset"])) / r["scale"], score=d["score"], text=t)
                 for d, t in zip(r["dets"], texts)]
        ev[r["cond"]].add(preds, r["gts"])
        gt_texts = predict(r["gt_crops"])
        gt_counts[r["cond"]][0] += sum(norm_text(p) == norm_text(t) for p, t in zip(gt_texts, r["texts"]))
        gt_counts[r["cond"]][1] += len(gt_texts)
    return {c: dict(e2e=ev[c].summary()["all"], gt_correct=gt_counts[c][0], gt_n=gt_counts[c][1],
                    gt_word_acc=gt_counts[c][0] / max(gt_counts[c][1], 1)) for c in ev}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--train-images", type=int, default=64)
    ap.add_argument("--val-images", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="results/archive/diagnostics/s3_audit/controlled_pilot")
    args = ap.parse_args()
    if min(args.steps, args.train_images, args.val_images, args.batch_size) < 1:
        ap.error("steps, image counts, and batch size must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config("configs/base.yaml", {"model": {"pretrained": False}})
    model = SeeRead(build_recognizer(cfg.rec), cfg).to(dev).eval()
    det_state = torch.load(DET, map_location="cpu", weights_only=False)
    init = torch.load(INIT, map_location="cpu", weights_only=False)
    model.det.load_state_dict(det_state["det"])
    model.det.requires_grad_(False)
    train = SeeReadDataset(cfg.data.root, "data/annotations/totaltext_train_main.jsonl", train=False, size=cfg.data.size)
    val = SeeReadDataset(cfg.data.root, "data/annotations/totaltext_val.jsonl", train=False, size=cfg.data.size)
    assert not ({r["image"] for r in train.items} & {r["image"] for r in val.items}), "train/val overlap"
    train_items = cache(model, train, args.train_images, dev, 4321, True)
    val_records = cache(model, val, args.val_images, dev, 1234, False)
    if not train_items:
        raise RuntimeError("No eligible training crops")
    indices = np.random.default_rng(0).integers(0, len(train_items), size=(args.steps, args.batch_size))
    report = dict(protocol="fixed v2 detector and GT/pred crops; identical S2 CRNN initialization; "
                  "paired lit/dark inputs; same sampled batches, FP32 AdamW lr=1e-4, no scheduler; "
                  "joint arm trains enhancer+rec with rec+0.1enh+0.5CIC; independent arm rec only; "
                  "not a full S3 reproduction or converged ranking",
                  args=vars(args), det=DET, init=INIT, train_crops=len(train_items),
                  sample_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                  train_sources={s: sum(x[5] == s for x in train_items) for s in ("gt", "pred")},
                  train_conditions={s: sum(x[4] == s for x in train_items) for s in ("lit", "dark")}, arms={})
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    for arm in ("joint_enh_rec", "independent_rec"):
        rec = build_recognizer(cfg.rec).to(dev)
        rec.load_state_dict(init["rec"])
        enh = copy.deepcopy(model.enh).to(dev) if arm == "joint_enh_rec" else None
        if enh is not None:
            enh.load_state_dict(init["enh"])
        before = evaluate(rec, enh, val_records, dev, args.batch_size)
        params = list(rec.parameters()) + (list(enh.parameters()) if enh is not None else [])
        opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
        losses, started = [], time.monotonic()
        for step, idx in enumerate(indices):
            rec.train()
            batch = [train_items[int(i)] for i in idx]
            x = torch.stack([b[0] for b in batch]).float().to(dev)
            texts = [b[3] for b in batch]
            opt.zero_grad(set_to_none=True)
            if enh is not None:
                enh.train()
                z = torch.stack([b[2] for b in batch]).to(dev)
                y, prm = enh(x, z)
                # Evaluate teacher without changing BN; student is in train mode.
                rec.eval()
                with torch.no_grad():
                    teacher = rec.encode_features(torch.stack([b[1] for b in batch]).float().to(dev))
                rec.train()
                loss = rec.forward_loss(y, texts) + 0.1 * enhancer_regulariser(x, y, prm)
                loss = loss + 0.5 * cic_loss(rec.encode_features(y), teacher)
            else:
                loss = rec.forward_loss(x, texts)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss in {arm} step {step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 5.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"nonfinite gradients in {arm} step {step}")
            opt.step()
            losses.append(float(loss.detach()))
            if (step + 1) % 20 == 0:
                print(arm, "step", step + 1, "loss", losses[-1], flush=True)
        elapsed = time.monotonic() - started
        after = evaluate(rec, enh, val_records, dev, args.batch_size)
        report["arms"][arm] = dict(before=before, after=after, training_seconds=elapsed,
                                    updates=args.steps, loss_first=losses[0], loss_last=losses[-1])
        checkpoint = dict(rec=rec.state_dict(), detector_path=DET, init_path=INIT, protocol=report["protocol"])
        if enh is not None:
            checkpoint["enh"] = enh.state_dict()
        torch.save(checkpoint, root / f"{arm}.pt")
        (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(arm, json.dumps(report["arms"][arm]), flush=True)
    print("Saved", root / "summary.json", flush=True)


if __name__ == "__main__":
    main()
