"""Fixed original/dark validation and centerline comparison, with paired records.

Example: python tools/legacy/evaluate_alignment.py --config configs/legacy/s4_rec_v4.yaml \
  --ckpt results/archive/legacy/s4_rec_v4/best.pt --out results/alignment/new_eval.json
Existing output and records are never replaced by a new invocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from seeread.config import load_config
from seeread.data.datasets import SeeReadDataset, collate
from seeread.metrics.spotting import SpottingEvaluator, edit_distance, norm_text
from seeread.models.recognizer import build_recognizer
from seeread.models.rectify import rectify_strip
from seeread.models.seeread import SeeRead


def serializable(value):
    if isinstance(value, SimpleNamespace):
        return {k: serializable(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_progress(path, report):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(serializable(report), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def rec_summary(records):
    correct = edits = chars = 0
    for record in records:
        ref, pred = norm_text(record["text"]), norm_text(record["prediction"])
        correct += pred == ref
        edits += edit_distance(pred, ref)
        chars += max(len(ref), 1)
    return dict(n_words=len(records), n_correct=correct,
                word_acc=correct / max(len(records), 1), edits=edits,
                ref_chars=chars, cer=edits / max(chars, 1))


@torch.inference_mode()
def gt_crop_records(model, batch, image, chunk):
    """Evaluate the dataset's eligible GT instances, independent of det matching."""
    instances = batch["instances"][0]
    if not instances:
        return []
    context = model.det.illum_context(image, model.det(image)["c5"])
    records = []
    for start in range(0, len(instances), chunk):
        group = instances[start:start + chunk]
        crops = torch.cat([rectify_strip(
            image, torch.as_tensor(ins["centers"], device=image.device),
            torch.as_tensor(ins["heights"], device=image.device), model.crop_hw)
            for ins in group])
        enhanced, _ = model.enh(crops, context.expand(len(group), -1))
        predictions = model.rec.predict(enhanced)
        if len(predictions) != len(group):
            raise RuntimeError("Recognizer returned a different number of GT predictions")
        records.extend(dict(eligible_index=start + j, text=ins["text"], prediction=pred,
                            centers=ins["centers"], heights=ins["heights"])
                       for j, (ins, pred) in enumerate(zip(group, predictions)))
    return records


def evaluate_combination(model, dataset, condition, mode, count, device, chunk,
                         with_gt, record_stream, on_progress):
    model.cfg.model.centerline_mode = mode
    model.cfg.model.rec_infer_batch = chunk
    evaluator, gt_records = SpottingEvaluator(), []
    loader = DataLoader(Subset(dataset, range(count)), batch_size=1,
                        shuffle=False, num_workers=0, collate_fn=collate)
    started = time.monotonic()
    for n, batch in enumerate(loader, 1):
        with torch.inference_mode():
            image = batch["image"].to(device)
            predictions = model.infer(image)[0]
            scale, (ox, oy) = batch["scale"][0], batch["offset"][0]
            for prediction in predictions:
                prediction["poly"] = (np.asarray(prediction["poly"]) - [ox, oy]) / scale
            eligible = gt_crop_records(model, batch, image, chunk) if with_gt else None
        bin_name, truth = batch["meta"][0].get("illum_bin"), batch["gt_orig"][0]
        evaluator.add(predictions, truth, image_bin=bin_name)
        per_image = SpottingEvaluator()
        per_image.add(predictions, truth, image_bin=bin_name)
        if eligible is not None:
            gt_records.extend(eligible)
        record = dict(condition=condition, centerline_mode=mode, index=batch["index"][0],
                      image=dataset.items[batch["index"][0]]["image"],
                      predictions=predictions, gt=truth, native=per_image.summary(),
                      eligible_gt_crops=eligible)
        record_stream.write(json.dumps(serializable(record), ensure_ascii=False, allow_nan=False) + "\n")
        record_stream.flush()
        result = dict(images_evaluated=n, images_requested=count, dataset_images=len(dataset),
                      partial=(n < len(dataset)), completed=(n == count),
                      elapsed_seconds=time.monotonic() - started, native=evaluator.summary(),
                      gt_crops=rec_summary(gt_records) if with_gt else None)
        # Persist after every image: an interrupted combination remains usable and explicitly partial.
        on_progress(result)
    return result if count else dict(images_evaluated=0, partial=True, completed=True,
                                    native={}, gt_crops=None)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--ann", help="Defaults to config data.val_ann; never falls back to test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0, help="0 means the entire annotation file")
    parser.add_argument("--centerline-modes", nargs="+", choices=("legacy", "sliced"), default=["legacy", "sliced"])
    parser.add_argument("--conditions", nargs="+", choices=("original", "dark"), default=["original", "dark"])
    parser.add_argument("--gt-crops", action="store_true", help="Also evaluate eligible GT crops; slower")
    parser.add_argument("--chunk", type=int, default=8, help="Recognition inference batch (maximum 8)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.7)
    args = parser.parse_args()
    if args.max_images < 0 or not 1 <= args.chunk <= 8 or not 0 < args.gpu_memory_fraction <= 1:
        parser.error("Require max-images >= 0, chunk in [1,8], and memory fraction in (0,1]")
    if len(set(args.centerline_modes)) != len(args.centerline_modes) or len(set(args.conditions)) != len(args.conditions):
        parser.error("Conditions and centerline modes must not contain duplicates")
    return args


def main():
    args = parse_args()
    cfg = load_config(args.config)
    ann = args.ann or getattr(cfg.data, "val_ann", None)
    if not ann:
        raise ValueError("No validation annotation configured; specify --ann explicitly")
    output = Path(args.out)
    records_path = output.with_name(output.stem + ".records.jsonl")
    if output.exists() or records_path.exists():
        raise FileExistsError(f"Choose new output paths: {output}, {records_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive reservation occurs before loading weights or initializing a GPU.
    with output.open("x", encoding="utf-8") as stream:
        stream.write('{"status":"initializing","partial":true}\n')
    record_stream = records_path.open("x", encoding="utf-8")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    cfg.model.pretrained = False  # Every detector tensor comes from the checkpoint; no ImageNet download.
    cfg.model.rec_infer_batch = args.chunk
    source_files = sorted(set(ROOT.joinpath("seeread").rglob("*.py")) | {Path(__file__).resolve()})
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in source_files}
    report = dict(status="initializing", partial=True, config_path=str(Path(args.config).resolve()),
                  resolved_config=serializable(cfg), checkpoint=str(Path(args.ckpt).resolve()),
                  checkpoint_sha256=sha256(args.ckpt), annotation=str(Path(ann).resolve()),
                  annotation_sha256=sha256(ann), source_hashes=source_hashes,
                  seed=args.seed, device=str(device), torch_version=torch.__version__,
                  gpu_memory_fraction=args.gpu_memory_fraction,
                  records_path=str(records_path.resolve()),
                  protocol=dict(native="custom greedy IoU>=0.5 spotting; no lexicon; alphanumeric lowercase",
                                dark="eval_condition=dark: fixed online darkening of lit sources only, before resize",
                                gt_crops="same eligible GT geometry per image, independent of detector matching",
                                max_instances_per_image=16, min_text_height_pixels=4,
                                gt_crops_enabled=args.gt_crops, chunk=args.chunk,
                                max_images=args.max_images, centerline_modes=args.centerline_modes,
                                conditions=args.conditions), results={})
    write_progress(output, report)
    try:
        # Loading on CPU and discarding checkpoint before moving the model avoids GPU duplication.
        model = SeeRead(build_recognizer(cfg.rec), cfg)
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        report["checkpoint_training_config"] = serializable(state.get("cfg"))
        report["weights_provenance"] = {part: "checkpoint; strict=True" for part in ("det", "enh", "rec")}
        for part in ("det", "enh", "rec"):
            getattr(model, part).load_state_dict(state[part], strict=True)
        del state
        model.to(device).eval()
        report["status"] = "running"
        for condition in args.conditions:
            dataset = SeeReadDataset(cfg.data.root, ann, train=False, size=cfg.data.size,
                                     synth_mode=cfg.data.synth_mode, seed=args.seed,
                                     eval_condition=condition)
            count = min(len(dataset), args.max_images) if args.max_images else len(dataset)
            if not count:
                raise ValueError("Annotation file contains no images")
            report["results"][condition] = {}
            for mode in args.centerline_modes:
                def progress(result):
                    report["results"][condition][mode] = result
                    write_progress(output, report)
                evaluate_combination(model, dataset, condition, mode, count, device, args.chunk,
                                     args.gt_crops, record_stream, progress)
                print(json.dumps({"condition": condition, "mode": mode,
                                  **report["results"][condition][mode]}, ensure_ascii=False), flush=True)
        report["status"] = "complete"
        report["partial"] = any(v["partial"] for modes in report["results"].values() for v in modes.values())
        write_progress(output, report)
    except BaseException as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["partial"] = True
        report["error"] = f"{type(error).__name__}: {error}"
        write_progress(output, report)
        raise
    finally:
        record_stream.close()


if __name__ == "__main__":
    main()
