"""Freeze existing v5 detection manifests without downloads or model execution.

Small datasets: inspect every image path, image header, SHA256 and polygon.
SynthText: stream every annotation for count/hash/schema; inspect explicitly
sampled images only. Its large original manifest is referenced by SHA256.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import shutil
import tempfile
import time
from zipfile import ZipFile

from PIL import Image

DATASETS = {
    "latedspot_train": ("latedspot_train.jsonl", 900),
    "latedspot_val": ("latedspot_val.jsonl", 100),
    "latedspot_test": ("latedspot_test.jsonl", 500),
    "totaltext_train": ("totaltext_train_main.jsonl", 1154),
    "totaltext_val": ("totaltext_val.jsonl", 100),
    "totaltext_test": ("totaltext_test.jsonl", 300),
    "ctw1500_train_all": ("ctw1500_train_det.jsonl", 1000),
    "ctw1500_test": ("ctw1500_test.jsonl", 500),
}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def relative_path(value, root):
    if not isinstance(value, str):
        raise ValueError("Image path must be a string")
    rel = PurePosixPath(value)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts or "\\" in value:
        raise ValueError(f"Unsafe data path: {value}")
    path = root / value
    path.resolve().relative_to(root.resolve())
    return path


def polygon_stats(instance):
    """Check annotation shape and finite coordinates; preserve original geometry."""
    if not isinstance(instance.get("ignore"), bool):
        raise ValueError("Each instance requires an explicit boolean ignore")
    if not isinstance(instance.get("text"), str):
        raise ValueError("Each instance requires a text string (### may mean det-only)")
    poly = instance.get("poly")
    if not isinstance(poly, list) or len(poly) < 3:
        raise ValueError("Each polygon needs at least three vertices")
    for point in poly:
        if (not isinstance(point, (list, tuple)) or len(point) != 2 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in point)):
            raise ValueError("Polygon coordinates must be finite numeric pairs")
    area = abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(poly, poly[1:] + poly[:1]))) / 2
    return dict(ignored=int(instance["ignore"]),
                det_only=int(not instance["ignore"] and instance["text"] in ("###", "")),
                transcribed=int(not instance["ignore"] and instance["text"] not in ("###", "")),
                zero_signed_area=int(area <= 1e-6), odd_vertices=len(poly) % 2)


def image_fingerprint(value, root, cache):
    path = relative_path(value, root)
    if value not in cache:
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            width, height = image.size
            fmt = image.format
        if min(width, height) <= 0:
            raise ValueError(f"Invalid image dimensions: {path}")
        cache[value] = dict(path=value, sha256=sha256(path), bytes=path.stat().st_size,
                            width=width, height=height, format=fmt)
    return cache[value]


def inspect_manifest(path, root, cache, expected_rows, sample_images=None, seed=0, keep_rows=True):
    """All row/schema checks are streaming; sample_images limits file access only."""
    full = sample_images is None
    sample_indices = (None if full else set(random.Random(seed).sample(range(expected_rows),
                                                min(sample_images, expected_rows))))
    if sample_indices is not None and sample_images:
        sample_indices.update((0, expected_rows - 1))
    stats, row_digest, kept = Counter(), hashlib.sha256(), []
    scenes, image_paths, content_hashes, sample_records = set(), set(), set(), []
    with Path(path).open("rb") as stream:
        for raw in stream:
            row_digest.update(raw)
            if not raw.strip():
                stats["blank_lines"] += 1
                continue
            row = json.loads(raw)
            index = stats["rows"]
            stats["rows"] += 1
            relative_path(row["image"], root)
            if row.get("lit"):
                relative_path(row["lit"], root)
            if not isinstance(row.get("instances"), list):
                raise ValueError(f"Missing instances list: {path}:{index + 1}")
            for instance in row["instances"]:
                stats.update(polygon_stats(instance))
                stats["instances"] += 1
            if full or index in sample_indices:
                fingerprints = {}
                for field in ("image", "lit"):
                    if row.get(field):
                        fp = image_fingerprint(row[field], root, cache)
                        fingerprints[field] = fp
                        stats[f"{field}_paths_verified"] += 1
                # Bbox bounds are recorded; we never silently clip or drop official instances.
                width, height = fingerprints["image"]["width"], fingerprints["image"]["height"]
                for instance in row["instances"]:
                    if any(x < 0 or y < 0 or x > width or y > height for x, y in instance["poly"]):
                        stats["bbox_outside_image_in_checked_rows"] += 1
                if not full:
                    sample_records.append(dict(row_index=index, image=row["image"],
                                               sha256=fingerprints["image"]["sha256"],
                                               width=width, height=height))
                else:
                    for fp in fingerprints.values():
                        image_paths.add(fp["path"])
                        content_hashes.add(fp["sha256"])
            if full:
                scene = row.get("meta", {}).get("scene")
                if scene is not None:
                    scenes.add(str(scene))
                if keep_rows:
                    kept.append(row)
    if stats["rows"] != expected_rows:
        raise ValueError(f"Unexpected full manifest count {path}: {stats['rows']} != {expected_rows}; no filtered replacement allowed")
    report = dict(source_path=str(Path(path).resolve()), source_sha256=row_digest.hexdigest(),
                  source_bytes=Path(path).stat().st_size, counts=dict(stats),
                  annotation_schema_checked="all rows; finite vertices, explicit boolean ignore, text strings",
                  image_check_scope="all rows" if full else "sampled rows only; full image existence NOT verified",
                  image_check_details="path existence, format/dimensions from header, file-byte SHA256; no full pixel decode",
                  geometry_limit="finite coordinate/schema and signed-area checks only; not a complete polygon topology validity test",
                  sample_seed=seed if not full else None, image_sample=sample_records if not full else None,
                  meta_scene_values=sorted(scenes) if full else None)
    return report, kept, dict(paths=image_paths, hashes=content_hashes, scenes=scenes)


def ctw_split(rows, fingerprints, seed=0, dev_count=100):
    """Keep identical image contents together while meeting the requested count."""
    groups = defaultdict(list)
    for row in rows:
        groups[fingerprints[row["image"]]["sha256"]].append(row)
    order = sorted(groups, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest())
    selected, n = set(), 0
    for group in order:
        if n + len(groups[group]) <= dev_count:
            selected.add(group); n += len(groups[group])
        if n == dev_count:
            break
    if n != dev_count:
        raise ValueError("Cannot create exact CTW dev count without dividing a duplicate-content group")
    train, dev = [], []
    for row in rows:
        (dev if fingerprints[row["image"]]["sha256"] in selected else train).append(row)
    return train, dev


def pair_overlap(first, second):
    return dict(shared_paths=len(first["paths"] & second["paths"]),
                shared_content_hashes=len(first["hashes"] & second["hashes"]),
                shared_meta_scene_values=sorted(first["scenes"] & second["scenes"]))


def row_sets(rows, fingerprints):
    paths, hashes, scenes = set(), set(), set()
    for row in rows:
        for field in ("image", "lit"):
            if row.get(field):
                paths.add(row[field]); hashes.add(fingerprints[row[field]]["sha256"])
        scene = row.get("meta", {}).get("scene")
        if scene is not None:
            scenes.add(str(scene))
    return dict(paths=paths, hashes=hashes, scenes=scenes)


def quarantine_test_overlap(development, test_rows, fingerprints):
    """Exclude any train/dev image pair sharing bytes with the locked test set."""
    heldout = defaultdict(set)
    for row in test_rows:
        for field in ("image", "lit"):
            if row.get(field):
                heldout[fingerprints[row[field]]["sha256"]].add(row["image"])
    cleaned, quarantine = {}, []
    for split, rows in development.items():
        cleaned[split] = []
        for index, row in enumerate(rows):
            hashes = {fingerprints[row[key]]["sha256"] for key in ("image", "lit") if row.get(key)}
            matches = sorted({path for value in hashes for path in heldout.get(value, ())})
            if matches:
                quarantine.append(dict(row, _v5_quarantine=dict(
                    reason="image_bytes_identical_to_locked_test", original_split=split,
                    original_row_index=index, image_sha256=fingerprints[row["image"]]["sha256"],
                    matching_test_images=matches)))
            else:
                cleaned[split].append(row)
    return cleaned, quarantine


def write_rows(path, rows, replace=False):
    path = Path(path)
    target = path.with_name(path.name + ".quarantine-tmp") if replace else path
    with target.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    if replace:
        target.replace(path)


def quarantine_existing(out):
    """Explicit repair of v5 derived training data, preserving all original data."""
    provenance_path = out / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    fingerprint_path = out / "image_fingerprints.jsonl"
    if sha256(fingerprint_path) != provenance["image_fingerprints"]["sha256"]:
        raise ValueError("Fingerprint audit changed; run a new full preparation")
    with fingerprint_path.open() as stream:
        fingerprints = {r["path"]: r for r in map(json.loads, stream)}
    data = {}
    for split in ("train", "val", "test"):
        key = "totaltext_" + split
        path = out / (key + ".jsonl")
        if sha256(path) != provenance["frozen_manifests"][key]["sha256"]:
            raise ValueError(f"Frozen manifest changed since audit: {path}")
        with path.open() as stream:
            data[split] = [json.loads(line) for line in stream if line.strip()]
    clean, quarantined = quarantine_test_overlap({k: data[k] for k in ("train", "val")}, data["test"], fingerprints)
    if not quarantined:
        print("No train/dev byte overlap with locked TotalText test; no changes", flush=True)
        return
    quarantine_path = out / "totaltext_quarantine.jsonl"
    write_rows(quarantine_path, quarantined)
    for split in ("train", "val"):
        if len(clean[split]) == len(data[split]):
            continue
        key, path = "totaltext_" + split, out / ("totaltext_" + split + ".jsonl")
        write_rows(path, clean[split], replace=True)
        provenance["frozen_manifests"][key].update(rows=len(clean[split]), sha256=sha256(path))
    provenance["original_overlap_checks"] = provenance["overlap_checks"].copy()
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        aa, bb = clean.get(a, data[a]), clean.get(b, data[b])
        provenance["overlap_checks"][f"totaltext_{a}__totaltext_{b}"] = pair_overlap(
            row_sets(aa, fingerprints), row_sets(bb, fingerprints))
    provenance["quarantine"] = dict(path=str(quarantine_path.resolve()), rows=len(quarantined),
                                     sha256=sha256(quarantine_path),
                                     source_split_counts=dict(Counter(r["_v5_quarantine"]["original_split"] for r in quarantined)))
    provenance["derived_totaltext_counts"] = {**{k: len(v) for k, v in clean.items()}, "test": len(data["test"])}
    provenance["policy"]["totaltext"] = ("Original1154/100/300 preserved in source. V5 train/dev exclude exact image-byte overlaps with locked test; "
                                            "see derived_totaltext_counts and totaltext_quarantine.jsonl. Original test is unchanged.")
    provenance["quarantine_repair_script_sha256"] = sha256(__file__)
    temporary = provenance_path.with_name("provenance.quarantine-tmp.json")
    write_json(temporary, provenance)
    temporary.replace(provenance_path)
    print(json.dumps(dict(derived_counts=provenance["derived_totaltext_counts"],
                          quarantine=provenance["quarantine"]), indent=2), flush=True)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def mlt17_instances(text):
    """RRC MLT17: eight coordinates, script, then the unescaped transcription."""
    instances = []
    for number, line in enumerate(text.lstrip("\ufeff").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split(",", 9)
        if len(fields) != 10:
            raise ValueError(f"MLT17 annotation line {number}: expected 10 fields")
        coords = [float(value) for value in fields[:8]]
        instance = dict(poly=[coords[i:i + 2] for i in range(0, 8, 2)],
                        lang=fields[8], text=fields[9], ignore=fields[9] == "###")
        polygon_stats(instance)
        instances.append(instance)
    return instances


def prepare_mlt17(root, out):
    """Validate/extract the official 7200/1800 release without changing frozen v5 data."""
    import cv2

    source = root / "raw/MLT17"
    groups = {
        "train_images": [f"ch8_training_images_{i}.zip" for i in range(1, 9)],
        "train_gt": ["ch8_training_localization_transcription_gt_v2.zip"],
        "val_images": ["ch8_validation_images.zip"],
        "val_gt": ["ch8_validation_localization_transcription_gt_v2.zip"],
    }
    products = [out / f"mlt17_{split}.jsonl" for split in ("train", "dev")]
    products += [out / "mlt17_train.reference.json", out / "mlt17_image_fingerprints.jsonl"]
    if any(path.exists() for path in products):
        raise FileExistsError("MLT17 manifests already exist; preserve the audited release")
    archives = []
    # Read each member to EOF: zipfile checks its CRC before it becomes usable.
    # Publish the four directories only after all eleven archives pass.
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=source) as temporary:
        staging = Path(temporary)
        for group, names in groups.items():
            existing = source / group
            destination = staging / group
            destination.mkdir()
            seen = set()
            for name in names:
                path = source / name
                with ZipFile(path) as archive:
                    members = archive.infolist()
                    for member in members:
                        # This official release is flat; reject unexpected paths or collisions.
                        if Path(member.filename).name != member.filename or member.is_dir():
                            raise ValueError(f"Unexpected MLT17 archive path: {member.filename}")
                        if member.filename in seen:
                            raise ValueError(f"Duplicate MLT17 archive member: {member.filename}")
                        seen.add(member.filename)
                        if existing.exists():
                            with archive.open(member) as src, (existing / member.filename).open("rb") as dst:
                                for chunk in iter(lambda: src.read(1024 * 1024), b""):
                                    if chunk != dst.read(len(chunk)):
                                        raise ValueError(f"Extracted MLT17 file differs from archive: {member.filename}")
                                if dst.read(1):
                                    raise ValueError(f"Extracted MLT17 file has extra bytes: {member.filename}")
                        else:
                            with archive.open(member) as src, (destination / member.filename).open("xb") as dst:
                                shutil.copyfileobj(src, dst, length=1024 * 1024)
                archives.append(dict(file=name, bytes=path.stat().st_size, sha256=sha256(path),
                                     members=len(members), crc="all members passed"))
                print(json.dumps(archives[-1]), flush=True)
            if existing.exists() and {p.name for p in existing.iterdir()} != seen:
                raise ValueError(f"Unexpected files in extracted MLT17 directory: {existing}")
        for name in groups:
            if not (source / name).exists():
                (staging / name).rename(source / name)

    out.mkdir(parents=True, exist_ok=True)
    fingerprints, reports, sets, conversions = {}, {}, {}, []
    quarantined = []
    for official, split, expected in (("train", "train", 7200), ("val", "dev", 1800)):
        images = {p.stem: p for p in (source / f"{official}_images").iterdir()}
        labels = {p.stem.removeprefix("gt_"): p for p in (source / f"{official}_gt").iterdir()}
        if len(images) != expected or images.keys() != labels.keys():
            raise ValueError(f"MLT17 {official}: require {expected} unique image/GT pairs")
        rows = []
        for key in sorted(images, key=lambda name: int(name.removeprefix("img_"))):
            path = images[key]
            # OpenCV's image reader does not support GIF in this runtime. Keep
            # the originals and explicitly record a lossless RGB first-frame PNG.
            if path.suffix.lower() == ".gif":
                derived = source / f"{official}_rgb"
                derived.mkdir(exist_ok=True)
                converted = derived / (path.stem + ".png")
                with Image.open(path) as image:
                    frames = getattr(image, "n_frames", 1)
                    image.seek(0)
                    image.convert("RGB").save(converted)
                conversions.append(dict(source=str(path.relative_to(root)), source_sha256=sha256(path),
                                        derived=str(converted.relative_to(root)), frames=frames,
                                        policy="first GIF frame, RGB lossless PNG; original retained"))
                path = converted
            pixels = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if pixels is None:
                raise ValueError(f"MLT17 image cannot be decoded by training reader: {path}")
            row = dict(image=str(path.relative_to(root)), lit=None,
                       instances=mlt17_instances(labels[key].read_text(encoding="utf-8-sig")),
                       meta=dict(dataset="MLT17", official_split=official, image_id=key))
            image_fingerprint(row["image"], root, fingerprints)
            rows.append(row)
        if split == "dev":
            retained = []
            for row in rows:
                if fingerprints[row["image"]]["sha256"] in sets["train"]["hashes"]:
                    quarantined.append(dict(image=row["image"], sha256=fingerprints[row["image"]]["sha256"],
                                            reason="official_validation_image_bytes_also_in_training"))
                else:
                    retained.append(row)
            rows = retained
        manifest = out / f"mlt17_{split}.jsonl"
        write_rows(manifest, rows)
        report, _, coverage = inspect_manifest(manifest, root, fingerprints, len(rows), keep_rows=False)
        report["official_release_images"] = expected
        report["full_pixel_decode"] = "all images with cv2.imread(IMREAD_COLOR), same reader as training"
        reports[split], sets[split] = report, coverage
    overlap = pair_overlap(sets["train"], sets["dev"])
    if overlap["shared_paths"] or overlap["shared_content_hashes"]:
        raise ValueError(f"MLT17 train/dev overlap: {overlap}; requires an explicit split decision")
    fingerprint_path = out / "mlt17_image_fingerprints.jsonl"
    write_rows(fingerprint_path, [fingerprints[key] for key in sorted(fingerprints)])
    cross_dataset_overlap = {}
    # Pretraining must not silently expose a downstream held-out image.
    legacy_fingerprints = out / "image_fingerprints.jsonl"
    if legacy_fingerprints.is_file():
        legacy = {row["path"]: row["sha256"] for row in map(json.loads, legacy_fingerprints.read_text().splitlines())}
        for path in sorted(out.glob("*.jsonl")):
            if path.stem.startswith(("latedspot_", "ctw1500_", "totaltext_")) and path.stem.endswith(("_val", "_dev", "_test")):
                overlaps = [row["image"] for row in map(json.loads, path.read_text().splitlines())
                            if legacy.get(row["image"]) in sets["train"]["hashes"]]
                cross_dataset_overlap[path.stem] = overlaps
        if any(cross_dataset_overlap.values()):
            raise ValueError(f"MLT17 pretraining overlaps downstream held-out data: {cross_dataset_overlap}")
    frozen = {f"mlt17_{split}": dict(path=f"mlt17_{split}.jsonl", rows=report["counts"]["rows"],
                                     sha256=report["source_sha256"], role="train_or_development")
              for split, report in reports.items()}
    descriptor = dict(reports["train"], schema="v5_mlt17_release_v1",
                      status="validated_complete_release", usable_for_paper_recipe=True,
                      archives=archives, splits=reports, overlap=overlap, conversions=conversions,
                      frozen_manifests=frozen, quarantined_validation=quarantined,
                      cross_dataset_heldout_overlap=cross_dataset_overlap,
                      image_fingerprints=dict(path=str(fingerprint_path.resolve()), sha256=sha256(fingerprint_path)),
                      policy="All7200 official train retained; dev from official1800 validation minus exact-byte training duplicates; original files retained",
                      upstream_digest="local SHA256 and ZIP CRC recorded; no upstream signed digest supplied",
                      source_script_sha256=sha256(__file__))
    descriptor["source_path"] = "mlt17_train.jsonl"  # Relative to reference file in host and container.
    write_json(out / "mlt17_train.reference.json", descriptor)
    print(json.dumps(dict(status=descriptor["status"], counts={k: v["counts"] for k, v in reports.items()},
                          overlap=overlap, gif_conversions=len(conversions)), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations"))
    parser.add_argument("--out", type=Path, default=Path("data/annotations/v5/detection"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synth-sample", type=int, default=256)
    parser.add_argument("--prepare-mlt17", action="store_true",
                        help="Validate/extract all eleven MLT17 ZIPs and add independent train/dev manifests")
    parser.add_argument("--quarantine-known-overlap", action="store_true",
                        help="Repair only existing v5 TotalText train/dev byte overlap; preserve original files and locked test")
    args = parser.parse_args()
    if args.prepare_mlt17:
        prepare_mlt17(args.data_root, args.out)
        return
    if args.synth_sample < 1:
        parser.error("--synth-sample must be positive")
    if args.quarantine_known_overlap:
        quarantine_existing(args.out)
        return
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Preserving existing v5 data protocol: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    reports, rows, sets, cache = {}, {}, {}, {}
    for name, (filename, count) in DATASETS.items():
        report, data, coverage = inspect_manifest(args.annotations / filename, args.data_root, cache, count)
        reports[name], rows[name], sets[name] = report, data, coverage
        print(json.dumps({"dataset": name, "counts": report["counts"]}), flush=True)
    train, dev = ctw_split(rows.pop("ctw1500_train_all"), cache, args.seed)
    rows["ctw1500_train"], rows["ctw1500_dev"] = train, dev
    for name in ("ctw1500_train", "ctw1500_dev"):
        sets[name] = row_sets(rows[name], cache)
    inherited_totaltext_overlap = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inherited_totaltext_overlap[f"totaltext_{a}__totaltext_{b}"] = pair_overlap(
            sets["totaltext_" + a], sets["totaltext_" + b])
    clean, quarantined = quarantine_test_overlap(
        {split: rows["totaltext_" + split] for split in ("train", "val")}, rows["totaltext_test"], cache)
    for split, values in clean.items():
        rows["totaltext_" + split] = values
        sets["totaltext_" + split] = row_sets(values, cache)
    overlap = {}
    for prefix, split_names in (("latedspot", ("train", "val", "test")),
                                ("totaltext", ("train", "val", "test")),
                                ("ctw1500", ("train", "dev", "test"))):
        for index, split in enumerate(split_names):
            for other in split_names[index + 1:]:
                a, b = f"{prefix}_{split}", f"{prefix}_{other}"
                check = pair_overlap(sets[a], sets[b])
                overlap[f"{a}__{b}"] = check
                if check["shared_paths"]:
                    raise ValueError(f"Image path leakage: {a}, {b}")
    # No small/subsampled SynthText manifest is created or substituted.
    synth, _, _ = inspect_manifest(args.annotations / "synthtext_train.jsonl", args.data_root,
                                   cache, 858750, args.synth_sample, args.seed, keep_rows=False)
    synth["role"] = "full original SynthText pretraining manifest; reference by content hash"
    reports["synthtext_train"] = synth
    frozen = {}
    for name, data in rows.items():
        target = args.out / f"{name}.jsonl"
        with target.open("x", encoding="utf-8") as stream:
            for row in data:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        frozen[name] = dict(path=str(target.resolve()), rows=len(data), sha256=sha256(target),
                            role="final_test_registration_only" if name.endswith("_test") else "train_or_development")
    fingerprint_path = args.out / "image_fingerprints.jsonl"
    with fingerprint_path.open("x", encoding="utf-8") as stream:
        for path in sorted(cache):
            stream.write(json.dumps(cache[path]) + "\n")
    write_json(args.out / "synthtext_train.reference.json", synth)
    quarantine_path = args.out / "totaltext_quarantine.jsonl"
    write_rows(quarantine_path, quarantined)
    mlt_root = args.data_root / "raw/MLT17"
    mlt_files = sum(1 for p in mlt_root.rglob("*") if p.is_file()) if mlt_root.exists() else 0
    mlt = dict(path=str(mlt_root.resolve()), files=mlt_files,
               status="missing_prerequisite" if not mlt_files else "unverified_prerequisite",
               usable_for_paper_recipe=False,
               explanation="No validated full MLT17 manifest. Never substitute a filtered or invented subset.")
    write_json(args.out / "mlt17.prerequisite.json", mlt)
    converters = [Path("tools/convert_annotations.py"), Path("tools/convert_new/convert_ctw1500_textdet.py")]
    provenance = dict(schema="v5_detection_protocol_v1", seed=args.seed, datasets=reports,
                      frozen_manifests=frozen, overlap_checks=overlap,
                      original_overlap_checks=inherited_totaltext_overlap,
                      derived_totaltext_counts={**{k: len(v) for k, v in clean.items()}, "test": len(rows["totaltext_test"])},
                      quarantine=dict(path=str(quarantine_path.resolve()), rows=len(quarantined), sha256=sha256(quarantine_path),
                                      source_split_counts=dict(Counter(r["_v5_quarantine"]["original_split"] for r in quarantined))),
                      image_fingerprints=dict(path=str(fingerprint_path.resolve()), rows=len(cache), sha256=sha256(fingerprint_path)),
                      mlt17=mlt, source_script_sha256=sha256(__file__),
                      existing_converter_sha256={str(p): sha256(p) for p in converters if p.is_file()},
                      policy=dict(no_downloads=True, no_models_run=True, original_manifests_unchanged=True,
                                  test="LATeD/TotalText/CTW test registered for final use only; no model or hyperparameter evaluation performed",
                                  lat_ed="Inherited 900/100/500 image split; overlapping coarse meta.scene prefixes recorded, no claim of scene isolation",
                                  totaltext="Original1154/100/300 retained; v5 train/dev exclude exact-byte overlaps with locked test into quarantine; constant meta.scene is not a real scene identifier",
                                  ctw="Original 1000 training images split 900/100 by image-content SHA groups, stable seed0 ordering; official500 test unchanged",
                                  annotations="Keep complete source rows and polygons/ignore/text unchanged. ### plus ignore=false means detection-only positive",
                                  synthtext="All858750 annotations counted/hashed/schema-checked streaming; image files sampled only; reference full source without duplicating1GB or creating filtered replacement",
                                  paper_alignment="Paper-guided reimplementation protocol, not exact Seeing reproduction; MLT17 pretraining prerequisite remains missing and existing converted data are not independently authenticated against upstream dataset digests",
                                  geometry="Out-of-bounds/zero-area counts reported without silent filtering; polygon self-intersection topology not fully verified"),
                      elapsed_seconds=time.monotonic() - started)
    write_json(args.out / "provenance.json", provenance)
    print(json.dumps(dict(frozen_manifests=frozen, synthtext=synth["counts"], mlt17=mlt,
                          overlap_checks=overlap), indent=2), flush=True)


if __name__ == "__main__":
    main()
