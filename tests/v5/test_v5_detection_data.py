"""CPU-only tests of detection manifest audit, grouping and partial coverage."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.v5_prepare_detection import (ctw_split, inspect_manifest, polygon_stats,
                                       mlt17_instances, prepare_mlt17,
                                       quarantine_existing, quarantine_test_overlap,
                                       relative_path, sha256)


class DetectionPreparationTests(unittest.TestCase):
    def test_totaltext_keeps_uppercase_image_extension(self):
        from tools import convert_annotations as converter
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "raw/TotalText/Images/Train/img61.JPG"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (32, 32), "gray").save(image)
            gt = root / "raw/TotalText/txt_format/Train"
            gt.mkdir(parents=True)
            (gt / "poly_gt_img61.txt").write_text(
                "x: [[2 20 20 2]], y: [[2 2 14 14]], ornt: [u'h'], transcriptions: [u'TEXT']\n"
                "x: [[2 20 2]], y: [[2 2 14]], ornt: [u'h'], transcriptions: [u'TRI']\n"
                "x: [[2 20 20 2]], y: [[2 2 14 14]], ornt: [u\"h\\'\"], transcriptions: [u\\'ALL\\']\n"
                "x: [[2 20 20 2]], y: [[2 2 14 14]], ornt: [], transcriptions: [u'and']\n")
            out = root / "annotations"
            out.mkdir()
            with patch.object(converter, "qa_sheet"):
                converter.conv_totaltext(str(root / "raw"), str(out), str(root), "train")
            rows = [json.loads(line) for line in (out / "totaltext_train.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["image"], "raw/TotalText/Images/Train/img61.JPG")
            self.assertEqual(rows[0]["instances"][0]["text"], "TEXT")
            self.assertEqual([v["text"] for v in rows[0]["instances"]], ["TEXT", "TRI", "ALL", "and"])
            self.assertEqual(len(rows[0]["instances"][1]["poly"]), 3)

    def test_totaltext_detection_ignore_follows_source_orientation(self):
        from tools import convert_annotations as converter
        # Released img1061 has orientation '#' with text 'R'; img904 has
        # empty orientation with text '#'. Column6, not transcription, marks DC.
        for orientation, text, ignored in [("u'#'", "R", True), ("", "#", False)]:
            with self.subTest(orientation=orientation, text=text), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                image = root / "raw/TotalText/Images/Train/img1.jpg"
                image.parent.mkdir(parents=True)
                Image.new("RGB", (32, 32), "gray").save(image)
                gt = root / "raw/TotalText/txt_format/Train"
                gt.mkdir(parents=True)
                (gt / "poly_gt_img1.txt").write_text(
                    f"x: [[2 20 20 2]], y: [[2 2 14 14]], ornt: [{orientation}], transcriptions: [u'{text}']\n")
                out = root / "annotations"
                out.mkdir()
                with patch.object(converter, "qa_sheet"):
                    converter.conv_totaltext(str(root / "raw"), str(out), str(root), "train")
                instance = json.loads((out / "totaltext_train.jsonl").read_text())["instances"][0]
                self.assertEqual(instance["ignore"], ignored)
                self.assertEqual(instance.get("src", instance["text"]), text)
                if not ignored:
                    self.assertEqual(instance["text"], text)

    def test_mlt17_preserves_multilingual_commas_and_explicit_ignore(self):
        rows = mlt17_instances("\ufeff1,2,8,2,8,9,1,9,Arabic,مرحبا, العالم\r\n"
                               "1,2,8,2,8,9,1,9,Latin,###\r\n")
        self.assertEqual(rows[0]["text"], "مرحبا, العالم")
        self.assertEqual(rows[0]["lang"], "Arabic")
        self.assertFalse(rows[0]["ignore"])
        self.assertTrue(rows[1]["ignore"])
        self.assertEqual(rows[0]["poly"], [[1, 2], [8, 2], [8, 9], [1, 9]])
        with self.assertRaises(ValueError):
            mlt17_instances("1,2,8,2,8,9,1,9,Latin")

    def test_mlt17_archive_path_cannot_escape_extraction(self):
        from zipfile import ZipFile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw/MLT17"
            source.mkdir(parents=True)
            with ZipFile(source / "ch8_training_images_1.zip", "w") as archive:
                archive.writestr("../escape.jpg", "invalid")
            with self.assertRaisesRegex(ValueError, "archive path"):
                prepare_mlt17(root, root / "annotations")
            self.assertFalse((source / "train_images").exists())
            self.assertFalse((source / "escape.jpg").exists())

    def test_ctw_conversion_preserves_explicit_detection_ignore_without_transcription(self):
        from tools import convert_annotations as converter
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'raw/CTW1500/ctw1500'
            image = source / 'textdet_imgs/test/1001.jpg'
            image.parent.mkdir(parents=True)
            Image.new('RGB', (32, 32), 'gray').save(image)
            polygon = [2, 2, 20, 2, 20, 14, 2, 14]
            row = dict(img_path='textdet_imgs/test/1001.jpg', instances=[
                dict(polygon=polygon, ignore=True), dict(polygon=polygon, ignore=False)])
            (source / 'textdet_test.json').write_text(json.dumps(dict(data_list=[row])))
            out = root / 'annotations'
            out.mkdir()
            with patch.object(converter, 'qa_sheet'):
                converter.conv_ctw1500(str(root / 'raw'), str(out), str(root), 'test')
            converted = json.loads((out / 'ctw1500_test.jsonl').read_text())
            self.assertEqual([v['ignore'] for v in converted['instances']], [True, False])
            self.assertEqual([v['text'] for v in converted['instances']], ['###', '###'])

    def test_full_annotations_and_sampled_paths_are_distinguished(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index in range(20):
                path = root / f"{index}.png"
                Image.new("RGB", (8, 8), (index, 0, 0)).save(path)
                rows.append(dict(image=path.name, lit=None, instances=[dict(
                    poly=[[1, 1], [7, 1], [7, 7], [1, 7]], text="###", ignore=False)]))
            manifest = root / "source.jsonl"
            content = "".join(json.dumps(row) + "\n" for row in rows)
            manifest.write_text(content)
            report, kept, sets = inspect_manifest(manifest, root, {}, 20, sample_images=2, keep_rows=False)
            self.assertEqual(report["source_sha256"], hashlib.sha256(content.encode()).hexdigest())
            self.assertEqual(report["counts"]["rows"], 20)
            self.assertEqual(report["counts"]["instances"], 20)
            self.assertEqual(report["counts"]["det_only"], 20)
            self.assertLess(report["counts"]["image_paths_verified"], 20)
            self.assertIn("NOT verified", report["image_check_scope"])
            self.assertEqual(kept, [])
            self.assertEqual(sets["paths"], set())
            full, kept, sets = inspect_manifest(manifest, root, {}, 20)
            self.assertEqual(full["counts"]["image_paths_verified"], 20)
            self.assertEqual(len(sets["hashes"]), 20)
            self.assertEqual(kept, rows)
            with self.assertRaises(ValueError):
                inspect_manifest(manifest, root, {}, 19)

    def test_content_duplicate_images_stay_together(self):
        rows = [dict(image=f"{i}.png") for i in range(10)]
        fingerprints = {row["image"]: {"sha256": str(index)} for index, row in enumerate(rows)}
        fingerprints["1.png"]["sha256"] = fingerprints["0.png"]["sha256"]
        train, dev = ctw_split(rows, fingerprints, seed=0, dev_count=3)
        again = ctw_split(list(reversed(rows)), fingerprints, seed=0, dev_count=3)
        self.assertEqual(len(dev), 3)
        self.assertEqual({r["image"] for r in dev}, {r["image"] for r in again[1]})
        self.assertFalse({fingerprints[r["image"]]["sha256"] for r in train} &
                         {fingerprints[r["image"]]["sha256"] for r in dev})
        self.assertEqual(("0.png" in {r["image"] for r in train}),
                         ("1.png" in {r["image"] for r in train}))

    def test_schema_failure_and_geometry_warnings(self):
        instance = dict(poly=[[0, 0], [1, 1], [2, 2]], text="###", ignore=False)
        self.assertEqual(polygon_stats(instance)["zero_signed_area"], 1)
        self.assertEqual(polygon_stats(instance)["det_only"], 1)
        with self.assertRaises(ValueError):
            polygon_stats(dict(instance, ignore="false"))
        with self.assertRaises(ValueError):
            polygon_stats(dict(instance, poly=[[0, 0], [1, 1], [float("nan"), 2]]))
        for value in ("../escape", "/absolute", "sub/../../escape"):
            with self.assertRaises(ValueError):
                relative_path(value, Path("/tmp"))

    def test_locked_test_content_is_quarantined_from_train_and_dev(self):
        rows = {split: [dict(image=f"{split}_{i}.png", lit=None, instances=[])
                         for i in range(2)] for split in ("train", "val", "test")}
        fingerprints = {row["image"]: dict(path=row["image"], sha256=row["image"])
                        for group in rows.values() for row in group}
        fingerprints["train_0.png"]["sha256"] = fingerprints["val_1.png"]["sha256"] = "test_0.png"
        clean, quarantine = quarantine_test_overlap({k: rows[k] for k in ("train", "val")}, rows["test"], fingerprints)
        self.assertEqual([r["image"] for r in clean["train"]], ["train_1.png"])
        self.assertEqual([r["image"] for r in clean["val"]], ["val_0.png"])
        self.assertEqual(len(quarantine), 2)
        self.assertEqual(quarantine[0]["_v5_quarantine"]["matching_test_images"], ["test_0.png"])
        self.assertNotIn("_v5_quarantine", rows["train"][0])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fp = root / "image_fingerprints.jsonl"
            fp.write_text("".join(json.dumps(r) + "\n" for r in fingerprints.values()))
            provenance = dict(image_fingerprints=dict(sha256=sha256(fp)), frozen_manifests={},
                              overlap_checks={}, policy={})
            for split, values in rows.items():
                name = f"totaltext_{split}"
                path = root / (name + ".jsonl")
                path.write_text("".join(json.dumps(r) + "\n" for r in values))
                provenance["frozen_manifests"][name] = dict(rows=len(values), sha256=sha256(path))
            (root / "provenance.json").write_text(json.dumps(provenance))
            test_bytes = (root / "totaltext_test.jsonl").read_bytes()
            original = root / "original_train.jsonl"
            original.write_bytes((root / "totaltext_train.jsonl").read_bytes())
            original_bytes = original.read_bytes()
            quarantine_existing(root)
            changed = json.loads((root / "provenance.json").read_text())
            self.assertEqual(changed["derived_totaltext_counts"], dict(train=1, val=1, test=2))
            self.assertEqual(changed["quarantine"]["rows"], 2)
            self.assertEqual(changed["overlap_checks"]["totaltext_train__totaltext_test"]["shared_content_hashes"], 0)
            self.assertEqual((root / "totaltext_test.jsonl").read_bytes(), test_bytes)
            self.assertEqual(original.read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()
