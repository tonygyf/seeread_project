"""CPU-only integration: frozen data binding, real losses/decoder and exact resume."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("v5_detection_pilot", ROOT / "tools/v5_detection_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 8, 1)
        self.dropout = nn.Dropout2d(.25)
        self.head = nn.Conv2d(8, 8, 1)

    def forward(self, image, with_scm=False, mask=None):
        feature = self.dropout(F.relu(self.features(F.avg_pool2d(image, 4))))
        value = self.head(feature)
        result = dict(text=value[:, :2], center=value[:, 2:4],
                      height=F.softplus(value[:, 4:5]), angle=torch.tanh(value[:, 5:6]))
        if with_scm:
            result["scm_mask"] = value[:, 6:].sigmoid()
        return result


class PilotTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        fingerprints, registrations = [], {}
        for split, count in (("train", 3), ("val", 2)):
            rows = []
            for index in range(count):
                name = f"{split}_{index}.png"
                value = 20 + index * 30 + (100 if split == "val" else 0)
                cv2.imwrite(str(self.data / name), np.full((32, 48, 3), value, np.uint8))
                rows.append(dict(image=name, instances=[dict(poly=[[8, 8], [36, 8], [36, 24], [8, 24]],
                                                             text="###", ignore=False)],
                                 meta=dict(source="LATeD", illum_kind="real")))
                fingerprints.append(dict(path=name, sha256=pilot.sha256(self.data / name)))
            manifest = self.root / ("latedspot_" + split + ".jsonl")
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
            registrations["latedspot_" + split] = dict(path=str(manifest), rows=count,
                                                       sha256=pilot.sha256(manifest), role="train_or_development")
        fp = self.root / "image_fingerprints.jsonl"
        fp.write_text("".join(json.dumps(row) + "\n" for row in fingerprints))
        provenance = self.root / "provenance.json"
        provenance.write_text(json.dumps(dict(schema="v5_detection_protocol_v1", frozen_manifests=registrations,
                                              image_fingerprints=dict(sha256=pilot.sha256(fp)))))
        self.data_config = dict(root=str(self.data), provenance=str(provenance), image_fingerprints=str(fp),
                                train_manifest=str(self.root / "latedspot_train.jsonl"),
                                val_manifest=str(self.root / "latedspot_val.jsonl"), train_samples=3, val_samples=2)
        selected, self.audit = pilot.load_pilot_rows(self.data_config, 0)
        self.train_data = pilot.FixedDetectionDataset(selected["train"], self.data, (32, 32))
        self.val_data = pilot.FixedDetectionDataset(selected["val"], self.data, (32, 32))
        self.cfg = dict(train=dict(max_seconds=100, max_steps=4, lr=.001, lr_step_updates=2,
                                   lr_gamma=.5, amp=False, seed=0, batch_size=2, val_batch_size=1,
                                   grad_clip=5., max_amp_skips=2, eval_every=2), decoder={})

    def tearDown(self):
        self.temp.cleanup()

    def _run(self, name, steps, model=None, checkpoint=None):
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        cfg = copy.deepcopy(self.cfg)
        cfg["train"]["max_steps"] = steps
        random.seed(19); np.random.seed(19); torch.manual_seed(19)
        model = model or TinyDetector()
        state = pilot.run_training(model, self.train_data, self.val_data, cfg, directory,
                                   torch.device("cpu"), dict(data=self.audit), checkpoint=checkpoint)
        saved = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
        return model, state, saved, directory

    def test_real_data_contract_and_exact_save_resume(self):
        uninterrupted, _, expected, _ = self._run("whole", 4)
        _, state, first, path = self._run("resumed", 2)
        self.assertEqual(state["step"], 2)
        self.assertTrue((path / "best.pt").is_file())
        self.assertEqual(first["scheduler"]["last_epoch"], 2)
        resumed, state, actual, _ = self._run("resumed", 4, checkpoint=first)
        for name, value in uninterrupted.state_dict().items():
            torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(actual["state"]["order"], expected["state"]["order"])
        self.assertEqual(actual["state"]["position"], expected["state"]["position"])
        self.assertEqual(actual["scheduler"], expected["scheduler"])
        self.assertEqual(state["step"], 4)
        self.assertTrue(list(path.glob("val_step_*.jsonl")))
        self.assertEqual(self.train_data[0]["image"].shape, (3, 32, 32))
        self.assertEqual(self.train_data[0]["mask"].shape, (32, 32))

    def test_test_selection_and_modified_images_are_rejected(self):
        cfg = dict(self.data_config, val_manifest=str(self.root / "latedspot_test.jsonl"))
        with self.assertRaisesRegex(ValueError, "test selection"):
            pilot.load_pilot_rows(cfg, 0)
        image = self.data / "train_0.png"
        cv2.imwrite(str(image), np.zeros((32, 48, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, "image bytes changed"):
            pilot.load_pilot_rows(self.data_config, 0)

    def test_partial_validation_cannot_select_best(self):
        model = TinyDetector()
        calls = iter((0., 200.))
        output = self.root / "partial.jsonl"
        result = pilot.evaluate(model, self.val_data, torch.device("cpu"), 1, output, {}, 100,
                                clock=lambda: next(calls))
        self.assertFalse(result["complete"])
        self.assertEqual(result["evaluated"], 1)
        self.assertFalse(output.exists())
        self.assertTrue(Path(result["records"]).is_file())

    def test_legacy_or_changed_provenance_resume_is_rejected(self):
        _, _, checkpoint, _ = self._run("source", 2)
        with self.assertRaisesRegex(ValueError, "checkpoint format"):
            pilot.validate_resume(dict(format="old_seeread", det={}), self.cfg, {})
        with self.assertRaisesRegex(ValueError, "provenance changed"):
            pilot.validate_resume(checkpoint, self.cfg, dict(data="changed"))


if __name__ == "__main__":
    unittest.main()
