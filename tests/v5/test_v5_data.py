"""Tests for pinned word-crop preparation; no network or GPU required."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.v5_prepare_reading import (git_blob_sha1, safe_relative, split_manifests,
                                      valid_hub_file, verified_members)


def row(identifier, image_hash, lit_hash, split="train", source="LSTR/test"):
    return dict(id=f"{split}:{identifier}", image="dark/" + identifier,
                lit="lit/" + identifier if lit_hash else None, text=identifier,
                source=source, verified_scene=None,
                dataset=dict(original_split=split),
                sha256=dict(image=image_hash, lit=lit_hash))


class PreparationTests(unittest.TestCase):
    def test_pair_and_duplicate_grouping_holdout_quarantine(self):
        values = [row(str(i), f"dark{i}", f"lit{i}") for i in range(30)]
        values += [row("duplicate", "another-dark", "lit0"),
                   row("heldout_duplicate", "test-dark", "lit1", "test"),
                   row("real", "real-dark", None, "test_real")]
        first = split_manifests(values, seed=0, dev_fraction=.2)
        second = split_manifests(list(reversed(values)), seed=0, dev_fraction=.2)
        self.assertEqual({k: [r["id"] for r in v] for k, v in first.items()},
                         {k: [r["id"] for r in v] for k, v in second.items()})
        self.assertEqual([r["id"] for r in first["quarantine"]], ["train:1"])
        memberships = {r["id"]: split for split, rows in first.items() for r in rows}
        self.assertEqual(memberships["train:0"], memberships["train:duplicate"])
        self.assertEqual(len(first["test_real"]), 1)
        self.assertFalse(first["test_real"][0]["scene_group_verified"])
        self.assertNotIn("poly", first["test_real"][0])
        self.assertFalse({r["source_group"] for r in first["train"]} &
                         {r["source_group"] for r in first["dev"]})

    def test_verified_archive_reuse_preserves_modified_old_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local, target = root / "old", root / "v5"
            (local / "dark").mkdir(parents=True)
            (local / "dark/a.png").write_bytes(b"authentic-image")
            (local / "dark/b.png").write_bytes(b"modified-old-file")
            archive = root / "official.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("dark/a.png", b"authentic-image")
                zipped.writestr("dark/b.png", b"official-image-b")
            found, stats = verified_members(archive, {"dark/a.png", "dark/b.png"}, local, target)
            self.assertEqual(stats, dict(reused_verified_images=1, extracted_images=1))
            self.assertEqual(found["dark/a.png"]["path"], (local / "dark/a.png").resolve())
            self.assertEqual((local / "dark/b.png").read_bytes(), b"modified-old-file")
            self.assertEqual((target / "dark/b.png").read_bytes(), b"official-image-b")
            metadata = dict(size=archive.stat().st_size, blobId=git_blob_sha1(archive))
            self.assertTrue(valid_hub_file(archive, metadata))
            metadata["lfs"] = dict(sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
            self.assertTrue(valid_hub_file(archive, metadata))
            metadata["lfs"]["sha256"] = "bad"
            self.assertFalse(valid_hub_file(archive, metadata))
        for unsafe in ("../escape", "/absolute", "good/../../escape", "folder\\escape"):
            with self.assertRaises(ValueError):
                safe_relative(unsafe)

    def test_explicit_scene_mapping_groups_distinct_word_crops(self):
        values = [row(str(i), f"d{i}", f"l{i}") for i in range(6)]
        values[0]["verified_scene"] = values[1]["verified_scene"] = "sceneA"
        result = split_manifests(values, seed=0, dev_fraction=.5)
        actual = {r["id"]: (split, r) for split, rows in result.items() for r in rows}
        self.assertEqual(actual["train:0"][0], actual["train:1"][0])
        self.assertTrue(actual["train:0"][1]["scene_group_verified"])


if __name__ == "__main__":
    unittest.main()
