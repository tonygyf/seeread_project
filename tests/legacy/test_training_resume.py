"""CPU integration of the real training loop with a tiny stochastic model/dataset.

Run: python tests/legacy/test_training_resume.py
No project data, pretrained downloads or GPU allocations are required.
"""
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from seeread.config import _ns
from tools.legacy import train


class TinyDataset(Dataset):
    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        return 9

    def __getitem__(self, index):
        # Exercise replay of dataset RNG separately from dropout in the model.
        return dict(x=torch.tensor([index / 10 + random.random() / 10 + np.random.rand() / 10]))


def tiny_collate(items):
    return dict(x=torch.stack([item["x"] for item in items]),
                image=torch.zeros(len(items), 3, 2, 2), scale=[1] * len(items),
                offset=[(0, 0)] * len(items), gt_orig=[[]] * len(items), meta=[{}] * len(items))


class TinyModel(torch.nn.Module):
    def __init__(self, recognizer, cfg):
        super().__init__()
        self.cfg = cfg
        self.det = torch.nn.Linear(1, 2)
        self.enh = torch.nn.Linear(2, 2)
        self.rec = torch.nn.Linear(2, 1)

    def forward_train(self, batch, pred_geom_prob):
        x = self.enh(self.det(batch["x"]))
        x = torch.nn.functional.dropout(x, p=.25, training=True)
        return dict(rec=(self.rec(x) - .7).square().mean())

    def infer(self, images):
        return [[] for _ in images]


class TinyEvaluator:
    def add(self, *args, **kwargs):
        pass

    def summary(self):
        return {"all": {"e2e_F": .6}}


def tiny_mixture(*args, **kwargs):
    dataset = TinyDataset()
    return dataset, WeightedRandomSampler(torch.ones(9), 9, replacement=True), [9]


class ResumeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = dict(data=dict(root="unused", train_ann="unused", val_ann="unused", size=8,
                                  synth_mode="physical", workers=0), model={}, rec={},
                        train=dict(stage="joint", epochs=3, batch_size=2, lr=.01, weight_decay=.001,
                                   amp=False, seed=4, pred_geom_warmup=.5, val_every=0,
                                   init_det=None, init_rec=None, checkpoint_every_steps=0))

    def tearDown(self):
        self.tmp.cleanup()

    def run_training(self, name, options=(), cfg=None):
        config = copy.deepcopy(cfg or self.cfg)
        config["train"]["out_dir"] = str(self.root / name)
        with patch.object(train, "load_config", return_value=_ns(config)), \
             patch.object(train, "SeeRead", TinyModel), \
             patch.object(train, "build_recognizer", return_value=None), \
             patch.object(train, "SeeReadDataset", TinyDataset), \
             patch.object(train, "collate", tiny_collate), \
             patch.object(train, "SpottingEvaluator", TinyEvaluator), \
             patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(sys, "argv", ["train.py", "--config", "unused", *map(str, options)]):
            train.main()
        return train.load_checkpoint(self.root / name / "last.pt")

    def assert_same_training(self, first, second):
        for module in ("det", "enh", "rec"):
            for key in first[module]:
                self.assertTrue(torch.equal(first[module][key], second[module][key]), (module, key))
        self.assertEqual(first["scheduler"], second["scheduler"])
        self.assertEqual(first["state"]["global_step"], second["state"]["global_step"])
        for parameter, values in first["optimizer"]["state"].items():
            for key, value in values.items():
                self.assertTrue(torch.equal(value, second["optimizer"]["state"][parameter][key]))

    def test_mid_epoch_resume_matches_uninterrupted(self):
        full = self.run_training("full")
        safe = torch.load(self.root / "full/last.pt", weights_only=True)
        self.assertEqual(safe["format_version"], 2)
        split = self.run_training("split", ["--max-steps", 3])
        self.assertEqual(split["state"]["global_step"], 3)
        self.assertEqual(split["state"]["batch_in_epoch"], 3)
        resumed = self.run_training("split", ["--resume", self.root / "split/last.pt"])
        self.assert_same_training(full, resumed)
        self.assertEqual(resumed["state"]["epoch"], 3)

    def test_mixture_resume_matches_uninterrupted(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["data"].update(train_anns=["a"], mix_weights=[1], mix_samples_per_epoch=9)
        with patch("seeread.data.mixture.build_mixture", side_effect=tiny_mixture):
            full = self.run_training("mix_full", cfg=cfg)
            self.run_training("mix_split", ["--max-steps", 5], cfg=cfg)
            resumed = self.run_training("mix_split", ["--resume", self.root / "mix_split/last.pt"], cfg=cfg)
        self.assert_same_training(full, resumed)

    def test_worker_prefetch_resume_matches_uninterrupted(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["data"]["workers"] = 2
        full = self.run_training("workers_full", cfg=cfg)
        self.run_training("workers_split", ["--max-steps", 3], cfg=cfg)
        resumed = self.run_training("workers_split", ["--resume", self.root / "workers_split/last.pt"], cfg=cfg)
        self.assert_same_training(full, resumed)

    def test_legacy_requires_explicit_warm_start_and_protects_outputs(self):
        full = self.run_training("old")
        legacy = {key: full[key] for key in ("det", "enh", "rec", "cfg")}
        path = self.root / "legacy.pt"
        torch.save(legacy, path)
        with self.assertRaisesRegex(ValueError, "Legacy checkpoint"):
            self.run_training("reject", ["--resume", path])
        warm = self.run_training("warm", ["--resume", path, "--resume-mode", "weights-only", "--max-steps", 1])
        self.assertEqual(warm["state"]["global_step"], 1)
        self.assertTrue(warm["provenance"]["optimizer_reset"])
        self.assertIn("config", warm)
        with self.assertRaisesRegex(ValueError, "nonempty output"):
            self.run_training("old", ["--resume", path, "--resume-mode", "weights-only"])
        with self.assertRaisesRegex(ValueError, "nonempty output"):
            self.run_training("old")

    def test_partial_validation_does_not_select_best(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["train"].update(epochs=1, val_every=1)
        self.run_training("partial", ["--val-max-batches", 1], cfg=cfg)
        self.assertFalse((self.root / "partial/best.pt").exists())
        events = [json.loads(line) for line in (self.root / "partial/train_events.jsonl").read_text().splitlines()]
        validation = next(event for event in events if event["event"] == "validation")
        self.assertTrue(validation["partial"])
        self.run_training("allval", cfg=cfg)
        self.assertTrue((self.root / "allval/best.pt").exists())

    def test_wall_budget_checkpoint_can_resume(self):
        full = self.run_training("wall_full")
        ticks = iter(np.arange(0, 100, .1))
        with patch.object(train.time, "monotonic", side_effect=lambda: float(next(ticks))):
            short = self.run_training("wall", ["--max-seconds", .5])
        self.assertLess(short["state"]["global_step"], 12)
        resumed = self.run_training("wall", ["--resume", self.root / "wall/last.pt"])
        self.assert_same_training(full, resumed)

    def test_exact_resume_rejects_recipe_change(self):
        self.run_training("recipe", ["--max-steps", 2])
        cfg = copy.deepcopy(self.cfg)
        cfg["train"]["lr"] = .1
        with self.assertRaisesRegex(ValueError, "Exact resume requires"):
            self.run_training("recipe", ["--resume", self.root / "recipe/last.pt"], cfg=cfg)

    def test_requires_cuda_when_requested(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["train"]["require_cuda"] = True
        with self.assertRaisesRegex(RuntimeError, "refusing CPU fallback"):
            self.run_training("no_cuda", cfg=cfg)

    def test_early_stop_counts_full_validations_and_saves_state(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["train"].update(epochs=8, val_every=1, early_stop_patience=2)
        stopped = self.run_training("early", cfg=cfg)
        self.assertEqual(stopped["state"]["epoch"], 3)
        self.assertEqual(stopped["state"]["val_no_improve"], 2)
        self.assertEqual(stopped["state"]["stop_reason"], "early_stop")
        partial = self.run_training("early_partial", ["--val-max-batches", 1], cfg=cfg)
        self.assertEqual(partial["state"]["epoch"], 8)
        self.assertEqual(partial["state"]["val_no_improve"], 0)

    def test_nonfinite_loss_stops_before_update_with_healthy_checkpoint(self):
        def nonfinite(model, batch, pred_geom_prob):
            return {"rec": model.rec.weight.sum() * float("nan")}
        with patch.object(TinyModel, "forward_train", nonfinite):
            with self.assertRaisesRegex(FloatingPointError, "nonfinite_loss"):
                self.run_training("nan")
        healthy = train.load_checkpoint(self.root / "nan/last.pt")
        failed = train.load_checkpoint(self.root / "nan/failed.pt")
        self.assertEqual(healthy["state"]["global_step"], 0)
        self.assertNotIn("stop_reason", healthy["state"])
        self.assertEqual(failed["state"]["stop_reason"], "nonfinite_loss")

    def test_nonfinite_gradient_stops_before_update(self):
        original = TinyModel.forward_train
        def nonfinite_gradient(model, batch, pred_geom_prob):
            model.rec.weight.register_hook(lambda grad: grad * float("inf"))
            return original(model, batch, pred_geom_prob)
        with patch.object(TinyModel, "forward_train", nonfinite_gradient):
            with self.assertRaisesRegex(FloatingPointError, "nonfinite_gradient"):
                self.run_training("infgrad")
        failed = train.load_checkpoint(self.root / "infgrad/failed.pt")
        self.assertEqual(failed["state"]["global_step"], 0)
        self.assertEqual(failed["optimizer"]["state"], {})

    def test_amp_overflow_backs_off_then_recovers_without_advancing_schedule(self):
        original_forward = TinyModel.forward_train
        original_scaler = torch.amp.GradScaler
        calls = [0]

        class GradientOverflow(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value.clone()

            @staticmethod
            def backward(ctx, gradient):
                return gradient * float("inf")

        def overflow_first_two(model, batch, pred_geom_prob):
            calls[0] += 1
            losses = original_forward(model, batch, pred_geom_prob)
            if calls[0] <= 2:
                losses = {key: GradientOverflow.apply(value) for key, value in losses.items()}
            return losses

        def cpu_scaler(**kwargs):
            # Use PyTorch's real CPU GradScaler to exercise its unscale/update state machine.
            return original_scaler("cpu", enabled=True, init_scale=8.)

        cfg = copy.deepcopy(self.cfg)
        cfg["train"]["amp_overflow_patience"] = 3
        with patch.object(TinyModel, "forward_train", overflow_first_two), \
             patch.object(torch.amp, "GradScaler", side_effect=cpu_scaler):
            recovered = self.run_training("amp_recover", cfg=cfg)
        self.assertEqual(recovered["state"]["amp_skips"], 2)
        self.assertEqual(recovered["state"]["amp_consecutive_skips"], 0)
        self.assertEqual(recovered["state"]["global_step"], 10)
        self.assertEqual(recovered["scheduler"]["last_epoch"], 10)
        self.assertEqual(recovered["scaler"]["scale"], 2.)
        self.assertTrue(all(value["step"].item() == 10 for value in recovered["optimizer"]["state"].values()))
        events = [json.loads(line) for line in (self.root / "amp_recover/train_events.jsonl").read_text().splitlines()]
        skips = [event for event in events if event["event"] == "amp_overflow_skip"]
        self.assertEqual([event["global_step"] for event in skips], [0, 0])
        self.assertEqual([event["scale_after"] for event in skips], [4., 2.])
        self.assertEqual([event["lr"] for event in skips], [[.01], [.01]])

        calls[0] = 0
        ticks = iter(np.arange(0, 100, .1))
        with patch.object(TinyModel, "forward_train", overflow_first_two), \
             patch.object(torch.amp, "GradScaler", side_effect=cpu_scaler):
            with patch.object(train.time, "monotonic", side_effect=lambda: float(next(ticks))):
                during_calibration = self.run_training("amp_resume", ["--max-seconds", .35], cfg=cfg)
            self.assertEqual(during_calibration["state"]["amp_consecutive_skips"], 1)
            self.assertEqual(during_calibration["scaler"]["scale"], 4.)
            resumed = self.run_training("amp_resume", ["--resume", self.root / "amp_resume/last.pt"], cfg=cfg)
        self.assert_same_training(recovered, resumed)
        self.assertEqual(recovered["scaler"], resumed["scaler"])

        calls[0] = 0
        cfg["train"]["amp_overflow_patience"] = 2
        with patch.object(TinyModel, "forward_train", overflow_first_two), \
             patch.object(torch.amp, "GradScaler", side_effect=cpu_scaler):
            with self.assertRaisesRegex(FloatingPointError, "amp_overflow_limit"):
                self.run_training("amp_limit", cfg=cfg)
        failed = train.load_checkpoint(self.root / "amp_limit/failed.pt")
        healthy = train.load_checkpoint(self.root / "amp_limit/last.pt")
        self.assertEqual(failed["state"]["amp_consecutive_skips"], 2)
        self.assertEqual(failed["state"]["global_step"], 0)
        self.assertEqual(failed["optimizer"]["state"], {})
        self.assertEqual(failed["scheduler"], healthy["scheduler"])
        for module in ("det", "enh", "rec"):
            for key, value in healthy[module].items():
                self.assertTrue(torch.equal(value, failed[module][key]))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
