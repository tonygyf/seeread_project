"""Formal-stage CPU integration with real indexed data, losses and checkpoint I/O."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("v5_train_detection", ROOT / "tools/v5_train_detection.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


class TinyDetector(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.feature = nn.Conv2d(3, width, 1)
        self.dropout = nn.Dropout2d(.2)
        self.head = nn.Conv2d(width, 8, 1)
        self.provenance = dict(channels=width, backbone="fake CPU test backbone", assumptions=["contract only"],
                               attention={}, dsconv={})

    def forward(self, image, with_scm=False, mask=None):
        feature = self.dropout(F.relu(self.feature(F.avg_pool2d(image, 4))))
        value = self.head(feature)
        result = dict(text=value[:, :2], center=value[:, 2:4], height=F.softplus(value[:, 4:5]),
                      angle=torch.tanh(value[:, 5:6]))
        if with_scm:
            result["scm_mask"] = value[:, 6:].sigmoid()
        return result


class StageTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for split, count in (("train", 3), ("dev", 2)):
            rows = []
            for index in range(count):
                name = f"image_{split}_{index}.png"
                image = np.full((32, 48, 3), 40 + index * 15 + (70 if split == "dev" else 0), np.uint8)
                cv2.imwrite(str(self.root / name), image)
                rows.append(dict(image=name, instances=[dict(poly=[[8, 8], [36, 8], [36, 24], [8, 24]],
                                                             text="###", ignore=False)]))
            with (self.root / f"tiny_{split}.jsonl").open("w") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
        self.cfg = dict(stage=dict(name="synthtext", epochs=2, init_checkpoint=None),
                        data=dict(root=str(self.root), train_manifest=str(self.root / "tiny_train.jsonl"),
                                  dev_manifest=str(self.root / "tiny_dev.jsonl"), image_hw=[32, 32], feature_hw=[8, 8],
                                  index_cache_dir=str(self.root / "indices"),
                                  augment=dict(flip_p=.5, scale_crop_p=1., scale_min=.75, scale_max=1.5)),
                        train=dict(seed=7, lr=.001, lr_step_epochs=1, lr_gamma=.5, amp=False,
                                   batch_size=2, workers=0, max_seconds=100., max_steps=0,
                                   val_every_epochs=1, val_batch_size=1, grad_clip=5., log_every=1,
                                   save_every_updates=1), decoder={})
        self.data = trainer.datasets(self.cfg)
        self.provenance = dict(scope="fake CPU contract test", sources={"stable": "sha"},
                               data={key: trainer.stable_dataset_provenance(value.source.provenance)
                                     for key, value in self.data.items()})

    def tearDown(self):
        for dataset in self.data.values():
            dataset.source.__del__()
        self.temp.cleanup()

    def _run(self, name, max_steps=0, checkpoint=None):
        config = copy.deepcopy(self.cfg)
        config["train"]["max_steps"] = max_steps
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        random.seed(31); np.random.seed(31); torch.manual_seed(31)
        model = TinyDetector()
        state = trainer.run_stage(model, self.data["train"], self.data["dev"], config, directory,
                                  torch.device("cpu"), self.provenance, checkpoint=checkpoint)
        saved = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
        return model, state, saved, directory

    def test_mid_epoch_resume_matches_uninterrupted_epochs_and_lr(self):
        self.cfg["train"]["workers"] = 2
        model, completed, expected, _ = self._run("whole")
        _, stopped, first, directory = self._run("split", max_steps=3)
        self.assertEqual(stopped["completed_epochs"], 1)
        self.assertEqual(stopped["position"], 2)
        resumed, final, saved, _ = self._run("split", checkpoint=first)
        self.assertEqual(final["status"], "stage_complete")
        self.assertEqual(final["completed_epochs"], 2)
        self.assertEqual(saved["scheduler"]["last_epoch"], 2)
        self.assertEqual(saved["scheduler"], expected["scheduler"])
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
        for state in (saved, expected):
            self.assertEqual(state["optimizer"]["param_groups"][0]["lr"], .00025)
        self.assertTrue((directory / "best.pt").is_file())
        self.assertTrue(list(directory.glob("dev_epoch_*.jsonl")))

    def test_stage_inheritance_rejects_legacy_topology_and_incomplete_chain(self):
        _, _, saved, _ = self._run("parent")
        child = TinyDetector()
        record = trainer.inherit_weights(child, saved, "synthtext")
        self.assertTrue(record["completed"])
        with self.assertRaisesRegex(ValueError, "topology-compatible"):
            trainer.inherit_weights(TinyDetector(width=12), saved, "synthtext")
        with self.assertRaises(ValueError):
            trainer.inherit_weights(child, dict(format="legacy", det={}), "synthtext")
        unfinished = copy.deepcopy(saved)
        unfinished["state"]["completed_epochs"] = 1
        with self.assertRaisesRegex(ValueError, "finish every declared epoch"):
            trainer.inherit_weights(child, unfinished, "synthtext")
        self.assertFalse(trainer.inherit_weights(child, unfinished, "synthtext", allow_reduced=True)["completed"])

    def test_amp_overflow_retries_same_batch_without_committing_bn_or_sampler(self):
        model = TinyDetector()
        model.feature = nn.Sequential(model.feature, nn.BatchNorm2d(8))
        attempts = []

        def overflow_once(gradient):
            attempts.append(True)
            if len(attempts) in (1, 3):
                return torch.full_like(gradient, float("inf"))
            return gradient

        model.head.weight.register_hook(overflow_once)
        config = copy.deepcopy(self.cfg)
        config["train"].update(amp=True, max_steps=2, max_amp_skips=2)
        output = self.root / "amp_retry"
        output.mkdir()
        scaler = torch.amp.GradScaler("cpu")
        with patch.object(trainer.torch.amp, "GradScaler", return_value=scaler):
            state = trainer.run_stage(model, self.data["train"], self.data["dev"], config, output,
                                      torch.device("cpu"), self.provenance)
        events = [json.loads(row) for row in (output / "events.jsonl").read_text().splitlines()]
        attempts = [row for row in events if row["event"] == "attempt"]
        self.assertEqual([row["position"] for row in attempts], [0, 0, 2, 2])
        self.assertEqual(attempts[0]["indices"], attempts[1]["indices"])
        self.assertEqual(attempts[2]["indices"], attempts[3]["indices"])
        self.assertEqual(state["step"], 2)
        self.assertEqual(state["position"], 0)
        self.assertEqual(state["amp_skips"], 2)
        self.assertEqual(state["amp_skip_streak"], 0)
        self.assertEqual(model.feature[1].num_batches_tracked.item(), 2)
        saved = torch.load(output / "last.pt", weights_only=False)
        self.assertEqual(saved["scheduler"]["last_epoch"], 1)
        self.assertTrue(all(item["step"].item() == 2 for item in saved["optimizer"]["state"].values()))

        model = TinyDetector()
        model.head.weight.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
        output = self.root / "amp_exhausted"
        output.mkdir()
        scaler = torch.amp.GradScaler("cpu")
        with patch.object(trainer.torch.amp, "GradScaler", return_value=scaler):
            with self.assertRaisesRegex(FloatingPointError, "Consecutive AMP"):
                trainer.run_stage(model, self.data["train"], self.data["dev"], config, output,
                                  torch.device("cpu"), self.provenance)
        saved = torch.load(output / "last.pt", weights_only=False)
        self.assertEqual(saved["state"]["step"], 0)
        self.assertFalse(saved["optimizer"]["state"])

    def test_missing_mlt_blocks_full_pipeline_before_gpu(self):
        descriptor = self.root / "mlt17.prerequisite.json"
        descriptor.write_text(json.dumps(dict(status="missing_prerequisite", usable_for_paper_recipe=False)))
        configs = []
        for stage, epochs in (("synthtext", 2), ("mlt17", 100), ("target", 250)):
            cfg = copy.deepcopy(self.cfg)
            cfg["stage"].update(name=stage, epochs=epochs)
            cfg["train"]["out_dir"] = str(self.root / "v5" / stage)
            if stage == "mlt17":
                cfg["data"]["train_manifest"] = str(descriptor)
            configs.append(cfg)
        with self.assertRaisesRegex(FileNotFoundError, "explicitly unavailable"):
            trainer.preflight_pipeline(configs)

    def test_paper_pipelines_keep_augmentation_and_predecessors_consistent(self):
        for filename in ('seeing_pipeline_lated.yaml', 'seeing_pipeline_ctw1500.yaml'):
            pipeline = ROOT / 'configs/v5' / filename
            descriptor = trainer.plain(trainer.load_config(pipeline))
            configs = [trainer.plain(trainer.load_config(pipeline.parent / name))
                       for name in descriptor['stages']]
            self.assertEqual([cfg['stage']['name'] for cfg in configs], list(trainer.STAGES))
            self.assertIsNone(configs[0]['stage']['init_checkpoint'])
            for cfg in configs:
                self.assertEqual(cfg['data']['augment'], configs[0]['data']['augment'])
                self.assertEqual(cfg['data']['augment']['scale_crop_p'], 1.)
            for previous, current in zip(configs, configs[1:]):
                self.assertEqual(Path(current['stage']['init_checkpoint']),
                                 Path(previous['train']['out_dir']) / 'last.pt')

    def test_output_lock_and_existing_artifacts_are_protected(self):
        output = self.root / "v5" / "locked"
        with trainer.output_lock(output):
            (output / "sentinel").write_text("preserve")
            with self.assertRaisesRegex(RuntimeError, "Another process"):
                with trainer.output_lock(output):
                    pass
        with self.assertRaises(FileExistsError):
            with trainer.output_lock(output):
                pass
        with trainer.output_lock(output, output / "last.pt"):
            self.assertEqual((output / "sentinel").read_text(), "preserve")

    def test_resume_rejects_changed_manifest_sources(self):
        model, _, saved, _ = self._run("source", max_steps=1)
        with self.assertRaisesRegex(ValueError, "provenance changed"):
            trainer.validate_resume(saved, self.cfg, dict(self.provenance, sources="changed"),
                                    trainer.topology_signature(model))

    def test_nonfinite_loss_aborts_without_replacing_good_checkpoint(self):
        class Nonfinite(TinyDetector):
            def forward(self, *args, **kwargs):
                result = super().forward(*args, **kwargs)
                result["text"] = result["text"] * float("nan")
                return result
        output = self.root / "nonfinite"
        output.mkdir()
        model = Nonfinite()
        with self.assertRaises(FloatingPointError):
            trainer.run_stage(model, self.data["train"], self.data["dev"], self.cfg, output,
                              torch.device("cpu"), self.provenance)
        good = torch.load(output / "last.pt", weights_only=False)
        interrupted = torch.load(output / "interrupted.pt", weights_only=False)
        self.assertTrue(good["resume_safe"])
        self.assertEqual(good["state"]["step"], 0)
        self.assertFalse(interrupted["resume_safe"])
        with self.assertRaisesRegex(ValueError, "diagnostic only"):
            trainer.validate_resume(interrupted, self.cfg, self.provenance, trainer.topology_signature(model))
        self.assertEqual(json.loads((output / "status.json").read_text())["state"]["status"], "error_abort")

    def test_finite_gradient_entries_with_overflowing_norm_never_update_weights(self):
        output = self.root / "norm_overflow"
        output.mkdir()
        model = TinyDetector()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        hooks = [parameter.register_hook(lambda gradient: torch.full_like(gradient, 1e30))
                 for parameter in model.parameters()]
        try:
            with self.assertRaisesRegex(FloatingPointError, "Nonfinite detection gradients"):
                trainer.run_stage(model, self.data["train"], self.data["dev"], self.cfg, output,
                                  torch.device("cpu"), self.provenance)
        finally:
            for hook in hooks:
                hook.remove()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        saved = torch.load(output / "last.pt", weights_only=False)
        self.assertEqual(saved["state"]["step"], 0)
        self.assertFalse(saved["optimizer"]["state"])
        self.assertNotIn('"event": "update"', (output / "events.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
