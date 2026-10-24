"""CPU-only checks of paired evaluation and partial-result accounting."""
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.legacy.evaluate_alignment import evaluate_combination, rec_summary, write_progress


class TinyDataset:
    def __init__(self):
        self.items = [{"image": f"image_{i}.png"} for i in range(3)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        # Original GT coordinates differ from network coordinates so the test
        # catches evaluating network pixels against original annotations.
        return dict(image=torch.zeros(3, 8, 8), targets={"text": torch.zeros(2, 2)},
                    instances=[], meta={"illum_bin": "lit"}, scale=2.0,
                    offset=(2, 2), orig_size=(3, 3), index=index,
                    gt_orig=[dict(poly=[[0, 0], [2, 0], [2, 2], [0, 2]], text="STOP", ignore=False)])


class TinyModel:
    cfg = SimpleNamespace(model=SimpleNamespace())

    def infer(self, image):
        text = "Stop!" if self.cfg.model.centerline_mode == "sliced" else "wrong"
        return [[dict(poly=np.array([[2, 2], [6, 2], [6, 6], [2, 6]]), score=1., text=text)]]


class EvaluationTests(unittest.TestCase):
    def test_paired_records_coordinate_recovery_and_partial(self):
        dataset, model = TinyDataset(), TinyModel()
        records = io.StringIO()
        progress = []
        for mode in ("legacy", "sliced"):
            result = evaluate_combination(model, dataset, "original", mode, 2,
                                          torch.device("cpu"), 8, False, records,
                                          lambda r: progress.append(r))
            self.assertTrue(result["partial"])
            self.assertTrue(result["completed"])
            self.assertEqual(result["native"]["all"]["det_F"], 1.)
            self.assertEqual(result["native"]["all"]["e2e_F"], float(mode == "sliced"))
        parsed = [json.loads(line) for line in records.getvalue().splitlines()]
        self.assertEqual([r["index"] for r in parsed], [0, 1, 0, 1])
        self.assertEqual([r["image"] for r in parsed[:2]], [r["image"] for r in parsed[2:]])
        self.assertEqual(len(progress), 4)
        self.assertTrue(all(r["partial"] for r in progress))
        self.assertTrue(all(r["eligible_gt_crops"] is None for r in parsed))

    def test_full_and_rec_aggregation(self):
        result = evaluate_combination(TinyModel(), TinyDataset(), "dark", "sliced", 3,
                                      torch.device("cpu"), 8, False, io.StringIO(), lambda r: None)
        self.assertFalse(result["partial"])
        summary = rec_summary([dict(text="A-b", prediction="ab"), dict(text="CAT", prediction="CT")])
        self.assertEqual(summary["word_acc"], .5)
        self.assertEqual(summary["ref_chars"], 5)
        self.assertEqual(summary["cer"], .2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            write_progress(path, result)
            self.assertEqual(json.loads(path.read_text())["images_evaluated"], 3)


if __name__ == "__main__":
    unittest.main()
