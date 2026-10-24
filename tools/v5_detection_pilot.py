"""Bounded LATeD engineering sanity, NOT an official Seeing benchmark run.

CLI requires explicit logical cuda:0 and a cached public ImageNetV2 backbone.
Only frozen V5 LATeD train/val registrations are allowed; tests are never used
for selection. The reusable loop also accepts CPU fake models in contract tests.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import random
import sys
import time
from urllib.parse import urlparse

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from seeread.config import load_config
from seeread.metrics.spotting import SpottingEvaluator
from seeread.v5.seeing_data import targets_from_polygons
from seeread.v5.seeing_decode import decode
from seeread.v5.seeing_losses import detection_loss
from seeread.v5.training_safety import record_amp_rejection

FORMAT = "seeread_v5_detection_pilot_v1"
PURPOSE = "fixed-subset overfit engineering sanity; not official benchmark evaluation"


def plain(value):
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return plain(vars(value)) if hasattr(value, "__dict__") else value


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def source_hashes():
    paths = [Path(__file__), ROOT / "seeread/config.py", ROOT / "seeread/metrics/spotting.py"]
    paths += [ROOT / "seeread/v5" / name for name in
              ("seeing.py", "dsconv.py", "seeing_data.py", "seeing_losses.py", "seeing_decode.py", "training_safety.py")]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def runtime_versions():
    value = dict(torch=torch.__version__, cuda=torch.version.cuda,
                 cuda_allocator_config=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'))
    for name in ("torchvision", "einops", "numpy", "opencv-python", "shapely"):
        try:
            value[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            value[name] = None
    return value


def require_cuda(device):
    if device != "cuda:0":
        raise ValueError("This pilot CLI requires explicit logical cuda:0; the launcher selects the physical GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this pilot has no CPU fallback")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def public_backbone_provenance():
    from torchvision.models import ResNet50_Weights
    weights = ResNet50_Weights.IMAGENET1K_V2
    filename = Path(urlparse(weights.url).path).name
    path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Public ImageNetV2 weights must be cached before the pilot: {path}")
    digest = sha256(path)
    expected_prefix = filename.rsplit("-", 1)[1].split(".")[0]
    if not digest.startswith(expected_prefix):
        raise ValueError("Cached backbone bytes do not match the official torchvision weight hash")
    return dict(name="ResNet50_Weights.IMAGENET1K_V2", url=weights.url, path=str(path),
                sha256=digest, initialization="public ImageNet only; independent random detection heads")


def _safe_image(root, value):
    path = (root / value).resolve()
    path.relative_to(root.resolve())
    if Path(value).is_absolute() or not path.is_file():
        raise ValueError(f"Invalid or missing image path in frozen data: {value}")
    return path


def load_pilot_rows(data, seed):
    """Bind manifests to their frozen train/val roles and all image fingerprints."""
    root = Path(data["root"])
    provenance_path = Path(data["provenance"])
    protocol = json.loads(provenance_path.read_text())
    if protocol.get("schema") != "v5_detection_protocol_v1":
        raise ValueError("Pilot requires the frozen V5 detection protocol")
    fingerprint_path = Path(data["image_fingerprints"])
    if sha256(fingerprint_path) != protocol["image_fingerprints"]["sha256"]:
        raise ValueError("Image fingerprint manifest changed after preparation")
    with fingerprint_path.open() as stream:
        fingerprints = {row["path"]: row for row in map(json.loads, stream)}
    selected, all_hashes, audit = {}, {}, {}
    for split, key, field in (("train", "latedspot_train", "train_manifest"),
                              ("val", "latedspot_val", "val_manifest")):
        path = Path(data[field])
        registration = protocol["frozen_manifests"][key]
        if path.name != key + ".jsonl" or "test" in path.name.lower():
            raise ValueError(f"Only {key}.jsonl may supply the {split} split; test selection is prohibited")
        if registration["role"] != "train_or_development":
            raise ValueError("Test/evaluation-only registrations cannot select pilot weights")
        digest = sha256(path)
        if digest != registration["sha256"]:
            raise ValueError(f"Frozen {split} annotation hash mismatch")
        with path.open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if len(rows) != registration["rows"]:
            raise ValueError(f"Frozen {split} row count changed")
        for row in rows:
            if row.get("meta", {}).get("source") != "LATeD" or row.get("meta", {}).get("illum_kind") != "real":
                raise ValueError("Pilot accepts only original real LATeD images; no online synthetic darkening")
            if row["image"] not in fingerprints:
                raise ValueError("An image is absent from the frozen fingerprint manifest")
        all_hashes[split] = {fingerprints[row["image"]]["sha256"] for row in rows}
        count = int(data[split + "_samples"])
        if not 1 <= count <= len(rows):
            raise ValueError(f"Invalid fixed {split} sample count: {count}")
        indices = sorted(random.Random(seed + (0 if split == "train" else 1)).sample(range(len(rows)), count))
        selected[split] = []
        for index in indices:
            row = rows[index]
            image_path = _safe_image(root, row["image"])
            expected = fingerprints[row["image"]]["sha256"]
            if sha256(image_path) != expected:
                raise ValueError(f"Selected image bytes changed after protocol freeze: {row['image']}")
            selected[split].append(dict(row=row, row_index=index, image_sha256=expected))
        audit[split] = dict(path=str(path), sha256=digest, full_rows=len(rows), selected_rows=count,
                            indices=indices, images=[dict(image=item["row"]["image"], sha256=item["image_sha256"])
                                                   for item in selected[split]])
    if all_hashes["train"] & all_hashes["val"]:
        raise ValueError("LATeD train/val contain identical image bytes")
    audit.update(protocol_sha256=sha256(provenance_path), fingerprint_sha256=sha256(fingerprint_path),
                 no_test_selection=True, scope=PURPOSE,
                 scene_limit="Image-content split isolation only; no claim of scene isolation")
    return selected, audit


class FixedDetectionDataset:
    """Small original-image dataset, cached after deterministic direct stretching."""

    def __init__(self, selected, root, input_hw, feature_stride=4, center_shrink=.3):
        self.samples = []
        height, width = map(int, input_hw)
        if min(height, width) < 32 or height % 32 or width % 32 or feature_stride != 4:
            raise ValueError("Seeing pilot uses a stride-4 detector and input dimensions divisible by 32")
        for item in selected:
            row = item["row"]
            bgr = cv2.imread(str(_safe_image(Path(root), row["image"])), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Cannot decode selected image: {row['image']}")
            original_hw = bgr.shape[:2]
            rgb = cv2.cvtColor(cv2.resize(bgr, (width, height)), cv2.COLOR_BGR2RGB)
            scale = np.array([width / original_hw[1], height / original_hw[0]], np.float32)
            instances = [dict(instance, poly=(np.asarray(instance["poly"], np.float32) * scale))
                         for instance in row["instances"]]
            targets = targets_from_polygons(instances, (height, width),
                                            (height // feature_stride, width // feature_stride), center_shrink)
            scm_mask = np.zeros((height, width), np.uint8)
            for instance in instances:
                if not instance.get("ignore", False):
                    cv2.fillPoly(scm_mask, [np.rint(instance["poly"]).astype(np.int32)], 1)
            self.samples.append(dict(image=torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255,
                                     targets={key: torch.from_numpy(value) for key, value in targets.items()},
                                     mask=torch.from_numpy(scm_mask).long(), original_hw=original_hw,
                                     row=row, image_sha256=item["image_sha256"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def make_batch(dataset, indices, device):
    samples = [dataset[index] for index in indices]
    return dict(image=torch.stack([sample["image"] for sample in samples]).to(device),
                mask=torch.stack([sample["mask"] for sample in samples]).to(device),
                targets={key: torch.stack([sample["targets"][key] for sample in samples]).to(device)
                         for key in samples[0]["targets"]})


@torch.no_grad()
def evaluate(model, dataset, device, batch_size, output, decoder, deadline, clock=time.monotonic):
    model.eval()
    evaluator = SpottingEvaluator(iou_thr=.5)
    count = 0
    target = Path(output)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("x") as stream:
        for begin in range(0, len(dataset), batch_size):
            if clock() >= deadline:
                break
            indices = list(range(begin, min(begin + batch_size, len(dataset))))
            image = torch.stack([dataset[index]["image"] for index in indices]).to(device)
            prediction = model(image)
            for local, index in enumerate(indices):
                sample = dataset[index]
                maps = {key: prediction[key][local] for key in ("text", "center", "height", "angle")}
                if not all(torch.isfinite(value).all() for value in maps.values()):
                    raise FloatingPointError("Nonfinite detector output during validation")
                predictions = decode(maps, image_hw=sample["original_hw"], **decoder)
                predictions = [dict(item, text="") for item in predictions]
                # Score geometry only: transcription never participates in selection.
                gt = [dict(instance, text="###") for instance in sample["row"]["instances"]]
                evaluator.add(predictions, gt)
                record = dict(image=sample["row"]["image"], image_sha256=sample["image_sha256"],
                              original_hw=list(sample["original_hw"]),
                              predictions=[dict(poly=item["poly"].tolist(), score=item["score"]) for item in predictions],
                              ground_truth=[dict(poly=np.asarray(item["poly"]).tolist(), ignore=bool(item.get("ignore"))) for item in gt])
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                count += 1
            stream.flush()
    complete = count == len(dataset)
    destination = target if complete else target.with_name(target.stem + ".partial.jsonl")
    temporary.replace(destination)
    raw_metrics = evaluator.summary().get("all") if count else None
    metrics = ({key: raw_metrics[key] for key in ("n_gt", "det_P", "det_R", "det_F", "frac_missed")}
               if raw_metrics else None)
    return dict(complete=complete, evaluated=count, expected=len(dataset), records=str(destination),
                metrics=metrics,
                protocol="internal score-greedy polygon IoU>=0.5; NOT official benchmark evaluation")


def validate_resume(checkpoint, cfg, provenance):
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Only this independent V5 detection pilot checkpoint format can resume")
    if checkpoint.get("resume_safe") is not True:
        raise ValueError("Interrupted in-flight updates cannot resume exactly; use the last known good checkpoint")
    def recipe(value):
        value = plain(value)
        for name in ("max_steps", "max_seconds", "out_dir"):
            value["train"].pop(name, None)
        return value
    required = {"model", "optimizer", "scheduler", "scaler", "rng", "state", "provenance", "config"}
    if required - checkpoint.keys() or recipe(checkpoint["config"]) != recipe(cfg):
        raise ValueError("Resume requires complete state and the same model/data/optimizer recipe")
    if checkpoint["provenance"] != provenance:
        raise ValueError("Source, initialization, runtime or frozen manifest provenance changed")


def detection_update_attempt(model, batch, optimizer, scaler, train, device, state, event, indices):
    """One attempt; AMP overflow restores buffers and retries the same position.

    RNG is deliberately consumed by rejected forwards. Only small registered
    buffers (including every BN counter/statistic) are snapshotted, not weights.
    A successful return is a committed optimizer update; None is a healthy
    rollback boundary with reduced scale, zero gradients and unchanged sampler.
    """
    maximum = int(train.get("max_amp_skips", 16))
    if not 1 <= maximum <= 16:
        raise ValueError("max_amp_skips must be explicitly bounded in [1,16] (8 or16 recommended)")
    state["attempts"] = state.get("attempts", 0) + 1
    state.setdefault("amp_skips", 0)
    event("attempt", attempt=state["attempts"], position=state["position"], indices=indices,
          scale=scaler.get_scale())
    model.train()
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}

    def restore_buffers():
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                if name in buffers:
                    buffer.copy_(buffers[name])

    def skip(reason, backoff=True):
        restore_buffers()
        optimizer.zero_grad(set_to_none=True)
        old_scale = scaler.get_scale()
        if backoff:
            # Explicit update handles BOTH element overflow recorded by
            # unscale_ and finite elements whose aggregate norm overflows.
            scaler.update(new_scale=old_scale * scaler.get_backoff_factor())
            saved_scale = scaler.state_dict()
            saved_scale["_growth_tracker"] = 0
            scaler.load_state_dict(saved_scale)
        record_amp_rejection(state, maximum, event, attempt=state["attempts"], position=state["position"], indices=indices,
              reason=reason,
              scale_before=old_scale, scale_after=scaler.get_scale(),
              buffers_restored=True, rng_policy="forward RNG consumed; sampler position unchanged")

    optimizer.zero_grad(set_to_none=True)
    try:
        with torch.autocast(device.type, enabled=bool(train["amp"]) and device.type == "cuda"):
            prediction = model(batch["image"], with_scm=True, mask=batch["mask"])
            losses = detection_loss(prediction, batch["targets"])
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite detection loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        entries_finite = all(torch.isfinite(parameter.grad).all() for parameter in parameters if parameter.grad is not None)
        amp = bool(train["amp"]) and scaler.is_enabled()
        if not entries_finite:
            if not amp:
                raise FloatingPointError("Nonfinite detection gradients")
            skip("nonfinite_gradient_entries")
            return None
        norm = torch.nn.utils.clip_grad_norm_(parameters, float(train["grad_clip"]), error_if_nonfinite=False)
        if not torch.isfinite(norm):
            if not amp:
                raise FloatingPointError("Nonfinite detection gradients: aggregate norm overflow")
            skip("nonfinite_gradient_norm")
            return None
        before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < before:
            skip("scaler_rejected_update", backoff=False)
            return None
        state["amp_skip_streak"] = 0
        return dict(loss=float(loss.detach()), losses={k: float(v.detach()) for k, v in losses.items()},
                    gradient_norm=float(norm))
    except BaseException:
        # An exception after optimizer entry is not an exact-resume boundary;
        # caller preserves last.pt and writes interrupted.pt as diagnostic-only.
        restore_buffers()
        optimizer.zero_grad(set_to_none=True)
        raise


def run_training(model, train_data, val_data, cfg, out_dir, device, provenance,
                 checkpoint=None, clock=time.monotonic, budget_started=None, scaler=None):
    """Reusable bounded loop; CLI alone owns CUDA policy and output locking."""
    train = cfg["train"]
    started = clock() if budget_started is None else budget_started
    deadline = started + float(train["max_seconds"])
    opt = torch.optim.Adam(model.parameters(), lr=float(train["lr"]), weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=int(train["lr_step_updates"]),
                                               gamma=float(train["lr_gamma"]))
    scaler = scaler if scaler is not None else torch.amp.GradScaler("cuda", enabled=bool(train["amp"]) and device.type == "cuda")
    state = dict(step=0, epoch=0, position=0, order=list(range(len(train_data))), best_det_f=-1.,
                 amp_skips=0, elapsed_seconds=0., status="running", validation_pending=False,
                 validation_attempt=0, attempts=0)
    random.Random(int(train["seed"])).shuffle(state["order"])
    if checkpoint is not None:
        validate_resume(checkpoint, cfg, provenance)
        model.load_state_dict(checkpoint["model"], strict=True)
        opt.load_state_dict(checkpoint["optimizer"])
        for values in opt.state.values():
            for key, value in values.items():
                if torch.is_tensor(value) and key != "step":
                    values[key] = value.to(device)
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        state = dict(checkpoint["state"])
        state["status"] = "running"
        restore_rng(checkpoint["rng"])
    previous_elapsed = state["elapsed_seconds"]
    out_dir = Path(out_dir)

    def event(kind, **values):
        payload = dict(event=kind, step=state["step"], epoch=state["epoch"],
                       elapsed_seconds=clock() - started, **values)
        with (out_dir / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(payload, allow_nan=False) + "\n")
        print(json.dumps(payload, allow_nan=False), flush=True)

    def save(name="last.pt"):
        state["elapsed_seconds"] = previous_elapsed + clock() - started
        payload = dict(format=FORMAT, resume_safe=name != "interrupted.pt", model=model.state_dict(), optimizer=opt.state_dict(),
                       scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), rng=rng_state(),
                       state=state, config=cfg, provenance=provenance, purpose=PURPOSE)
        temporary = out_dir / (name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(out_dir / name)

    event("start", purpose=PURPOSE, train_images=len(train_data), val_images=len(val_data),
          device=str(device), resumed=checkpoint is not None, max_amp_skips=int(train.get("max_amp_skips", 16)),
          overflow_policy="backoff; rollback buffers; retry same batch; consume forward RNG")
    save()
    try:
        while state["step"] < int(train["max_steps"]) or state["validation_pending"]:
            if clock() >= deadline:
                state["status"] = "budget_stop"
                break
            if state["validation_pending"]:
                state["validation_attempt"] += 1
                save()
                report = evaluate(model, val_data, device, int(train["val_batch_size"]),
                                  out_dir / f"val_step_{state['step']:06d}_attempt_{state['validation_attempt']:03d}.jsonl",
                                  cfg["decoder"], deadline, clock)
                event("validation", **report)
                if report["complete"]:
                    state["validation_pending"] = False
                    score = report["metrics"]["det_F"]
                    if score > state["best_det_f"]:
                        state["best_det_f"] = score
                        save("best.pt")
                        event("best", det_F=score, selection="complete fixed validation only")
                else:
                    state["status"] = "budget_stop_partial_validation"
                save()
                if not report["complete"]:
                    break
                continue
            if state["position"] >= len(train_data):
                state["epoch"] += 1
                state["position"] = 0
                state["order"] = list(range(len(train_data)))
                random.Random(int(train["seed"]) + state["epoch"]).shuffle(state["order"])
            begin = state["position"]
            indices = state["order"][begin:begin + int(train["batch_size"])]
            batch = make_batch(train_data, indices, device)
            result = detection_update_attempt(model, batch, opt, scaler, train, device, state, event, indices)
            if result is None:
                save()  # healthy rollback: same batch, advanced RNG, reduced scaler
                continue
            state["position"] += len(indices)
            state["step"] += 1
            scheduler.step()
            event("update", **result, lr=opt.param_groups[0]["lr"])
            if state["step"] % int(train["eval_every"]) == 0:
                state["validation_pending"] = True
                save()
        if state["status"] == "running":
            state["status"] = "max_steps_stop"
        save()
        event("stop", status=state["status"], best_det_f=state["best_det_f"])
    except (RuntimeError, FloatingPointError, ValueError, OSError, MemoryError, KeyboardInterrupt) as error:
        state["status"] = ("resource_abort" if isinstance(error, (torch.cuda.OutOfMemoryError, MemoryError))
                           or "out of memory" in str(error).lower() else "error_abort")
        event("abort", status=state["status"], error_type=type(error).__name__, message=str(error))
        # Preserve a complete interrupted state when model tensors remain finite;
        # never overwrite the last known good checkpoint with corrupt weights.
        if all(torch.isfinite(value).all() for value in model.state_dict().values()):
            save("interrupted.pt")
        raise
    finally:
        write_json(out_dir / "status.json", dict(state=state, purpose=PURPOSE,
                                                total_invocation_seconds=clock() - started))
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", type=Path, help="Trusted checkpoint from this same independent pilot")
    args = parser.parse_args()
    budget_started = time.monotonic()
    cfg = plain(load_config(args.config))
    train, data = cfg["train"], cfg["data"]
    if not (0 < float(train["memory_fraction"]) <= .35 and 0 < float(train["max_seconds"]) <= 1200):
        raise ValueError("Pilot resource ceilings are memory_fraction<=0.35 and max_seconds<=1200")
    for field in ("max_steps", "batch_size", "val_batch_size", "eval_every", "lr_step_updates"):
        if int(train[field]) < 1:
            raise ValueError(f"Positive {field} is required")
    if int(cfg["detector"]["channels"]) != 256 or cfg["detector"].get("pretrained") is not True:
        raise ValueError("The real pilot requires 256 channels and public ImageNet pretrained=True")
    device = require_cuda(train["device"])
    torch.cuda.set_per_process_memory_fraction(float(train["memory_fraction"]), 0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cv2.setNumThreads(1)
    random.seed(train["seed"]); np.random.seed(train["seed"]); torch.manual_seed(train["seed"])
    out_dir = Path(train["out_dir"])
    if "v5" not in out_dir.parts:
        raise ValueError("Use an independent V5 output directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / ".pilot.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This pilot output directory is already locked by an active process") from error
        existing = [path for path in out_dir.iterdir() if path.name != ".pilot.lock"]
        if existing and args.resume is None:
            raise FileExistsError("Preserving existing V5 artifacts; select a fresh output directory or explicit resume")
        if args.resume is not None:
            args.resume.resolve().relative_to(out_dir.resolve())
        selected, audit = load_pilot_rows(data, int(train["seed"]))
        train_data = FixedDetectionDataset(selected["train"], data["root"], data["input_hw"],
                                           data["feature_stride"], data["center_shrink"])
        val_data = FixedDetectionDataset(selected["val"], data["root"], data["input_hw"],
                                         data["feature_stride"], data["center_shrink"])
        backbone = public_backbone_provenance()
        from seeread.v5.seeing import build_detector
        model = build_detector(cfg["detector"]).to(device)
        provenance = dict(data=audit, backbone=backbone, detector=model.provenance,
                          source_hashes=source_hashes(), runtime=runtime_versions(), device=str(device),
                          cuda_name=torch.cuda.get_device_name(0),
                          config_path=str(Path(args.config)),
                          config_sha256=sha256(args.config),
                          training="Adam 1e-4; StepLR counted in updates for bounded pilot, not paper epochs",
                          evaluation="fixed LATeD validation, internal diagnostic det_F; not official benchmark",
                          pretraining_scope="This bounded target-only pilot does not run SynthText/MLT pretraining")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
        if checkpoint is not None:
            validate_resume(checkpoint, cfg, provenance)
        write_json(out_dir / "config.json", cfg)
        write_json(out_dir / "provenance.json", provenance)
        run_training(model, train_data, val_data, cfg, out_dir, device, provenance, checkpoint,
                     budget_started=budget_started)


if __name__ == "__main__":
    main()
