"""Independent epoch-based Seeing-style detection stages, with strict provenance.

The full pipeline preflights every prerequisite before starting SynthText. A
single stage needs --allow-reduced-data and is explicitly not a complete paper
reproduction. No old SeeRead/recognizer/enhancer checkpoints are accepted.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import json
import os
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from seeread.config import load_config
from seeread.v5.seeing_dataset import SeeingDataset
from v5_detection_pilot import (detection_update_attempt, evaluate, plain, public_backbone_provenance, require_cuda,
                                restore_rng, rng_state, runtime_versions, sha256, write_json)

FORMAT = "seeread_v5_detection_stage_v1"
STAGES = ("synthtext", "mlt17", "target")


def stable_dataset_provenance(value):
    value = copy.deepcopy(value)
    for key in ("index_cache_hit", "index"):
        value.pop(key, None)
    return value


def source_hashes():
    paths = [Path(__file__), ROOT / "tools/v5_detection_pilot.py", ROOT / "seeread/config.py",
             ROOT / "seeread/metrics/spotting.py"]
    paths += [ROOT / "seeread/v5" / name for name in
              ("seeing.py", "dsconv.py", "seeing_data.py", "seeing_losses.py", "seeing_decode.py", "seeing_dataset.py", "training_safety.py")]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def topology_signature(model):
    provenance = model.provenance
    snake = provenance.get("dsconv", {})
    return dict(channels=provenance.get("channels"), attention=provenance.get("attention"),
                assumptions=provenance.get("assumptions"), backbone=provenance.get("backbone"),
                implementation_sha256={name: sha256(ROOT / "seeread/v5" / name)
                                       for name in ("seeing.py", "dsconv.py")},
                geometry="direct full feature height; pi-periodic tangent angle",
                dsconv={key: snake.get(key) for key in
                        ("variant", "source_sha256", "kernel_size", "extend_scope", "if_offset")},
                shapes={key: list(value.shape) for key, value in model.state_dict().items()})


def preflight_pipeline(configs):
    """Check the full named chain before GPU allocation; never skip missing MLT."""
    if [cfg["stage"]["name"] for cfg in configs] != list(STAGES):
        raise ValueError("Pipeline must explicitly contain SynthText -> MLT17 -> target stages")
    for cfg in configs:
        stage = cfg["stage"]
        expected_epochs = {"synthtext": 2, "mlt17": 100, "target": 250}[stage["name"]]
        if int(stage["epochs"]) != expected_epochs:
            raise ValueError("Full staged recipe requires epochs 2/100/250; target250 is an explicit interpretation")
        path = Path(cfg["data"]["train_manifest"])
        if not path.is_file():
            raise FileNotFoundError(f"{stage['name']} prerequisite missing: {path}; no stage is silently skipped")
        if path.suffix != ".jsonl":
            descriptor = json.loads(path.read_text())
            if descriptor.get("usable_for_paper_recipe") is False or descriptor.get("status") == "missing_prerequisite":
                raise FileNotFoundError(f"{stage['name']} prerequisite explicitly unavailable: {path}")
            if stage["name"] == "mlt17" and descriptor.get("usable_for_paper_recipe") is not True:
                raise FileNotFoundError(f"MLT17 prerequisite remains unverified: {path}")
            if {"source_path", "source_sha256", "source_bytes"} - descriptor.keys():
                raise ValueError(f"Training reference is incomplete: {path}")
            source = Path(descriptor["source_path"])
            if not source.is_absolute():
                source = path.parent / source
            if not source.is_file():
                source = Path(cfg["data"]["root"]) / "annotations" / source.name
            if not source.is_file() or source.stat().st_size != descriptor["source_bytes"] or sha256(source) != descriptor["source_sha256"]:
                raise ValueError(f"Full training reference cannot be verified: {path}")
        if not Path(cfg["data"]["dev_manifest"]).is_file():
            raise FileNotFoundError(f"Development manifest missing: {cfg['data']['dev_manifest']}")
    if configs[0]["stage"].get("init_checkpoint"):
        raise ValueError("SynthText begins from public ImageNet plus independent random heads")
    for previous, current in zip(configs, configs[1:]):
        expected = Path(previous["train"]["out_dir"]) / "last.pt"
        if Path(current["stage"]["init_checkpoint"]) != expected:
            raise ValueError("Pipeline stage initialization must name the preceding stage's last.pt")


class StageDatasetView:
    """Adapt the indexed image dataset to a detection-only SCM/records contract."""

    def __init__(self, source):
        self.source = source

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        sample = self.source[index]
        h, w = sample["image"].shape[-2:]
        mask = np.zeros((h, w), np.uint8)
        for instance in sample["transformed_instances"]:
            if not instance.get("ignore", False):
                cv2.fillPoly(mask, [np.rint(instance["poly"]).astype(np.int32)], 1)
        fingerprint = self.source.fingerprints.get(sample["image_path"], {})
        return dict(image=sample["image"], targets=sample["targets"], mask=torch.from_numpy(mask).long(),
                    original_hw=sample["original_hw"], image_sha256=fingerprint.get("sha256"),
                    row=dict(image=sample["image_path"], instances=sample["original_instances"]))


def collate_stage(samples):
    return dict(image=torch.stack([sample["image"] for sample in samples]),
                mask=torch.stack([sample["mask"] for sample in samples]),
                targets={key: torch.stack([sample["targets"][key] for sample in samples])
                         for key in samples[0]["targets"]})


class EpochBatches:
    """Explicit saved permutation/cursor; workers receive deterministic epoch IDs."""

    def __init__(self, order, position, epoch, batch_size):
        self.order, self.position, self.epoch, self.batch_size = order, position, epoch, batch_size

    def __iter__(self):
        for start in range(self.position, len(self.order), self.batch_size):
            yield [(self.epoch, int(index)) for index in self.order[start:start + self.batch_size]]

    def __len__(self):
        return (len(self.order) - self.position + self.batch_size - 1) // self.batch_size


def datasets(cfg):
    data, train = cfg["data"], cfg["train"]
    result = {}
    for role, field in (("train", "train_manifest"), ("dev", "dev_manifest")):
        source = SeeingDataset(
            data[field], root=data["root"], split=role, image_hw=data["image_hw"], feature_hw=data["feature_hw"],
            illumination="original", augment=data.get("augment") if role == "train" else None,
            seed=int(train["seed"]), center_shrink=float(data.get("center_shrink", .3)),
            index_cache_dir=data.get("index_cache_dir"), image_fingerprints=data.get("image_fingerprints"),
            verify_fingerprints=True, require_fingerprint=bool(data.get("require_fingerprint", False)),
            protocol_provenance=data.get("protocol_provenance"),
        )
        result[role] = StageDatasetView(source)
    train_source, dev_source = result["train"].source, result["dev"].source
    if train_source.provenance["manifest_sha256"] == dev_source.provenance["manifest_sha256"]:
        raise ValueError("Training and development cannot use the same manifest")
    if train_source.split != "train" or dev_source.split != "dev":
        raise ValueError("Training/development roles are invalid")
    # Stream source row paths (not pixels or an in-memory SynthText row list).
    # Fingerprints are only checked where the prepared registry covers them.
    development_paths, development_hashes = set(), set()
    for index in range(len(dev_source)):
        row = dev_source._row(index)
        development_paths.add(row["image"])
        digest = dev_source.fingerprints.get(row["image"], {}).get("sha256")
        if digest:
            development_hashes.add(digest)
    with train_source.manifest.open("rb") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            image = json.loads(raw)["image"]
            known_digest = train_source.fingerprints.get(image, {}).get("sha256")
            if image in development_paths or (known_digest and known_digest in development_hashes):
                raise ValueError("Training/development overlap in image paths or known image bytes")
    result["train"].source.provenance["development_isolation"] = dict(
        image_paths_checked="all source annotation rows, streaming", known_hashes_checked=True,
        full_image_hash_coverage_claimed=False)
    return result


def inherit_weights(model, checkpoint, expected_stage, allow_reduced=False):
    if checkpoint.get("format") != FORMAT or checkpoint.get("topology") != topology_signature(model):
        raise ValueError("Stage inheritance accepts only topology-compatible V5 detection-stage checkpoints")
    if checkpoint.get("resume_safe") is not True:
        raise ValueError("An in-flight interrupted checkpoint is not a safe stage initializer")
    if checkpoint["config"]["stage"]["name"] != expected_stage:
        raise ValueError("Wrong predecessor stage in checkpoint")
    complete = int(checkpoint["state"]["completed_epochs"]) >= int(checkpoint["config"]["stage"]["epochs"])
    if not complete and not allow_reduced:
        raise ValueError("Full pipeline requires the predecessor stage to finish every declared epoch")
    model.load_state_dict(checkpoint["model"], strict=True)
    return dict(predecessor=expected_stage, completed=complete,
                completed_epochs=checkpoint["state"]["completed_epochs"],
                inherited_checkpoint_provenance=checkpoint["provenance"])


def _recipe(cfg):
    cfg = copy.deepcopy(cfg)
    for key in ("max_seconds", "max_steps", "out_dir"):
        cfg["train"].pop(key, None)
    return cfg


def validate_resume(checkpoint, cfg, provenance, topology):
    needed = {"model", "optimizer", "scheduler", "scaler", "rng", "sampler", "loader_rng", "state", "provenance", "config", "topology"}
    if checkpoint.get("format") != FORMAT or needed - checkpoint.keys():
        raise ValueError("Exact resume requires a complete V5 detection-stage checkpoint")
    if checkpoint.get("resume_safe") is not True:
        raise ValueError("Interrupted in-flight updates are diagnostic only; resume the last known good checkpoint")
    if checkpoint["topology"] != topology or _recipe(checkpoint["config"]) != _recipe(cfg):
        raise ValueError("Topology or stage recipe differs from the saved training run")
    if checkpoint["provenance"] != provenance:
        raise ValueError("Source/runtime/dataset/initialization provenance changed; exact resume refused")


def run_stage(model, train_data, dev_data, cfg, out_dir, device, provenance, checkpoint=None,
              clock=time.monotonic, budget_started=None):
    """Epoch semantics and full sampler resume; CPU is solely for injected test models."""
    started = clock() if budget_started is None else budget_started
    train, stage = cfg["train"], cfg["stage"]
    deadline = started + float(train["max_seconds"]) if train["max_seconds"] else float("inf")
    out = Path(out_dir)
    topology = topology_signature(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train["lr"]), weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(train["lr_step_epochs"]),
                                               gamma=float(train["lr_gamma"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train["amp"]) and device.type == "cuda")
    state = dict(step=0, epoch=0, position=0, completed_epochs=0, best_det_f=-1.,
                 validation_pending=False, validation_attempt=0, elapsed_seconds=0., status="running")
    order, loader_rng = None, None
    if checkpoint is not None:
        validate_resume(checkpoint, cfg, provenance, topology)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value) and key != "step":
                    values[key] = value.to(device)
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        state = dict(checkpoint["state"])
        state["status"] = "running"
        order = checkpoint["sampler"]["order"]
        loader_rng = checkpoint["loader_rng"]
        restore_rng(checkpoint["rng"])
    prior_elapsed = state["elapsed_seconds"]

    def event(kind, **fields):
        row = dict(event=kind, stage=stage["name"], step=state["step"], epoch=state["epoch"],
                   elapsed_seconds=clock() - started, **fields)
        with (out / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)

    def save(name="last.pt"):
        state["elapsed_seconds"] = prior_elapsed + clock() - started
        payload = dict(format=FORMAT, resume_safe=name != "interrupted.pt", model=model.state_dict(), optimizer=optimizer.state_dict(),
                       scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), rng=rng_state(),
                       sampler=dict(order=order, position=state["position"], epoch=state["epoch"]),
                       loader_rng=loader_rng, state=state, config=cfg, provenance=provenance, topology=topology)
        temporary = out / (name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(out / name)

    def exhausted():
        return clock() >= deadline or (int(train["max_steps"]) > 0 and state["step"] >= int(train["max_steps"]))

    event("start", images=len(train_data), development_images=len(dev_data), resumed=checkpoint is not None,
          purpose="paper-guided detection stage; internal diagnostic evaluator, not official benchmark")
    save()
    try:
        while state["epoch"] < int(stage["epochs"]) or state["validation_pending"]:
            if state["position"] >= len(train_data):
                scheduler.step()
                state["epoch"] += 1
                state["completed_epochs"] = state["epoch"]
                state["position"] = 0
                order = None
                state["validation_pending"] = state["epoch"] % int(train["val_every_epochs"]) == 0
                event("epoch_end", completed_epochs=state["completed_epochs"], lr=optimizer.param_groups[0]["lr"])
                save()
            if state["validation_pending"]:
                if clock() >= deadline:
                    state["status"] = "budget_stop_validation_pending"
                    break
                state["validation_attempt"] += 1
                save()
                report = evaluate(model, dev_data, device, int(train["val_batch_size"]),
                                  out / f"dev_epoch_{state['epoch']:04d}_attempt_{state['validation_attempt']:03d}.jsonl",
                                  cfg["decoder"], deadline, clock)
                event("development", **report)
                if report["complete"]:
                    state["validation_pending"] = False
                    score = report["metrics"]["det_F"]
                    if score > state["best_det_f"]:
                        state["best_det_f"] = score
                        save("best.pt")
                        event("best", det_F=score, selection="complete development only")
                else:
                    state["status"] = "budget_stop_partial_development"
                save()
                if not report["complete"]:
                    break
                continue
            if state["epoch"] >= int(stage["epochs"]):
                break
            if exhausted():
                state["status"] = "budget_stop"
                break
            if order is None:
                order = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(
                    int(train["seed"]) + state["epoch"]))
            loader_generator = torch.Generator().manual_seed(int(train["seed"]) + 100000 + state["epoch"])
            batches = EpochBatches(order, state["position"], state["epoch"], int(train["batch_size"]))
            loader = DataLoader(train_data, batch_sampler=batches, num_workers=int(train["workers"]),
                                collate_fn=collate_stage, generator=loader_generator)
            iterator = iter(loader)
            loader_rng = loader_generator.get_state()
            while state["position"] < len(train_data):
                if exhausted():
                    state["status"] = "budget_stop"
                    break
                batch = next(iterator)
                batch["image"], batch["mask"] = batch["image"].to(device), batch["mask"].to(device)
                batch["targets"] = {key: value.to(device) for key, value in batch["targets"].items()}
                batch_size = batch["image"].shape[0]
                indices = order[state["position"]:state["position"] + batch_size].tolist()
                # Retain the materialized batch across AMP retries; advancing
                # the DataLoader would silently change samples/augmentations.
                update = None
                while update is None and not exhausted():
                    update = detection_update_attempt(model, batch, optimizer, scaler, train, device,
                                                      state, event, indices)
                if update is None:
                    state["status"] = "budget_stop"
                    break
                state["step"] += 1
                state["position"] += batch_size
                if state["step"] % int(train["log_every"]) == 0:
                    event("update", **update, lr=optimizer.param_groups[0]["lr"], position=state["position"])
                if state["step"] % int(train["save_every_updates"]) == 0:
                    save()
            del iterator, loader
            if state["status"] == "budget_stop" and state["position"] < len(train_data):
                break
        if state["status"] == "running":
            state["status"] = "stage_complete"
        save()
        event("stop", status=state["status"], completed_epochs=state["completed_epochs"])
    except (RuntimeError, ValueError, OSError, FloatingPointError, MemoryError, KeyboardInterrupt) as error:
        state["status"] = "resource_abort" if isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError)) or "out of memory" in str(error).lower() else "error_abort"
        event("abort", status=state["status"], error_type=type(error).__name__, message=str(error))
        if all(torch.isfinite(value).all() for value in model.state_dict().values()):
            save("interrupted.pt")
        raise
    finally:
        write_json(out / "status.json", dict(state=state, scope=provenance.get("scope"),
                                            total_invocation_seconds=clock() - started))
    return state


@contextmanager
def output_lock(out_dir, resume=None):
    out = Path(out_dir)
    if "v5" not in out.parts:
        raise ValueError("Detection stages require an independent V5 output directory")
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".stage.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process holds this output directory") from error
        if any(path.name != ".stage.lock" for path in out.iterdir()) and resume is None:
            raise FileExistsError("Existing stage artifacts are preserved; use explicit resume or a new directory")
        if resume:
            Path(resume).resolve().relative_to(out.resolve())
        yield out


def execute(cfg, *, resume=None, reduced=False, public_only=False):
    began = time.monotonic()
    stage, train, data = cfg["stage"], cfg["train"], cfg["data"]
    if stage["name"] not in STAGES or int(stage["epochs"]) < 1:
        raise ValueError("Unknown or empty detection stage")
    if not 0 < float(train["memory_fraction"]) <= .35 or float(train["max_seconds"]) < 0 or int(train["max_steps"]) < 0:
        raise ValueError("Invalid resource budget; memory_fraction cannot exceed0.35")
    for field in ("batch_size", "val_batch_size", "lr_step_epochs", "val_every_epochs", "save_every_updates", "log_every"):
        if int(train[field]) < 1:
            raise ValueError(f"Positive {field} is required")
    if int(train["workers"]) < 0 or not np.isfinite(train["grad_clip"]) or float(train["grad_clip"]) <= 0:
        raise ValueError("workers must be nonnegative and grad_clip must be finite and positive")
    if cfg["detector"].get("pretrained") is not True or cfg["detector"]["channels"] != 256:
        raise ValueError("Real stages require public ImageNetV2 and256channels; reduced CPU widths are test-only")
    if public_only and not reduced:
        raise ValueError("Public-only target/MLT initialization is allowed only as an explicit reduced-data stage")
    with output_lock(train["out_dir"], resume) as out:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False) if resume else None
        if checkpoint is not None:
            if checkpoint.get("format") != FORMAT:
                raise ValueError("Only independent V5 detection-stage checkpoints can resume")
            # Continuing an existing full stage preserves its scientific scope;
            # a standalone CLI invocation never retroactively changes its label.
            reduced = checkpoint["provenance"]["scope"].startswith("independent_reduced_data_stage")
        data_sets = datasets(cfg)  # Index/preflight before CUDA; do not materialize all annotations.
        device = require_cuda(train["device"])
        torch.cuda.set_per_process_memory_fraction(float(train["memory_fraction"]), 0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        cv2.setNumThreads(1)
        random.seed(train["seed"]); np.random.seed(train["seed"]); torch.manual_seed(train["seed"])
        backbone = public_backbone_provenance()
        from seeread.v5.seeing import build_detector
        model = build_detector(cfg["detector"])
        initialization = dict(kind="public_ImageNetV2_random_detection_heads", backbone=backbone)
        init_path = None if public_only else stage.get("init_checkpoint")
        if stage["name"] == "synthtext" and init_path:
            raise ValueError("The first SynthText stage must initialize from public ImageNet only")
        if stage["name"] != "synthtext" and not init_path and not reduced:
            raise ValueError("Full recipe requires the declared predecessor stage checkpoint")
        if init_path:
            inherited = torch.load(init_path, map_location="cpu", weights_only=False)
            expected_stage = STAGES[STAGES.index(stage["name"]) - 1]
            initialization = inherit_weights(model, inherited, expected_stage, allow_reduced=reduced)
            initialization.update(kind="inherited_V5_detection_stage", path=init_path, sha256=sha256(init_path))
            del inherited
        model.to(device)
        provenance = dict(scope="independent_reduced_data_stage_NOT_complete_reproduction" if reduced else "declared_full_staged_recipe_reconstruction_NOT_author_implementation",
                          stage=stage["name"], initialization=initialization,
                          data={key: stable_dataset_provenance(value.source.provenance) for key, value in data_sets.items()},
                          source_hashes=source_hashes(), runtime=runtime_versions(), detector=model.provenance,
                          device=str(device), cuda_name=torch.cuda.get_device_name(0),
                          assumptions=["target250epochs interprets an ambiguous paper training duration",
                                       "augmentation probabilities, resize and topology are documented reimplementation choices",
                                       "internal development det_F is not official benchmark evaluation",
                                       "CUDA grid_sample backward may not be bitwise deterministic across runs"])
        if checkpoint is not None:
            validate_resume(checkpoint, cfg, provenance, topology_signature(model))
        write_json(out / "config.json", cfg)
        write_json(out / "provenance.json", provenance)
        return run_stage(model, data_sets["train"], data_sets["dev"], cfg, out, device, provenance,
                         checkpoint, budget_started=began)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path)
    mode.add_argument("--pipeline", type=Path)
    parser.add_argument("--allow-reduced-data", action="store_true")
    parser.add_argument("--public-only", action="store_true", help="Explicitly start an independent stage from public ImageNet")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.config:
        if not args.allow_reduced_data:
            parser.error("A single stage needs --allow-reduced-data; use --pipeline for the preflighted complete chain")
        execute(plain(load_config(args.config)), resume=args.resume, reduced=True, public_only=args.public_only)
        return
    if args.allow_reduced_data or args.public_only:
        parser.error("Pipeline mode forbids reduced/public-only shortcuts")
    descriptor = plain(load_config(args.pipeline))
    paths = [args.pipeline.parent / value for value in descriptor["stages"]]
    configs = [plain(load_config(path)) for path in paths]
    preflight_pipeline(configs)  # Missing MLT blocks before any GPU/model operation.
    start = 0
    if args.resume:
        matches = [index for index, cfg in enumerate(configs)
                   if args.resume.resolve().parent == Path(cfg["train"]["out_dir"]).resolve()]
        if len(matches) != 1:
            raise ValueError("Pipeline resume checkpoint must belong to exactly one configured stage directory")
        start = matches[0]
    for index, cfg in enumerate(configs[start:], start=start):
        state = execute(cfg, resume=args.resume if index == start else None)
        if state["status"] != "stage_complete":
            print("Stage stopped at its resource budget; downstream stages will not start", flush=True)
            break


if __name__ == "__main__":
    main()
