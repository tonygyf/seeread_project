"""Prepare paired TotalText development or held-out spotting images with human GT."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from seeread.data.lowlight_synth import PHYS_LEVELS, darken, illum_bin, luminance


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def encoded_rgb(path):
    """Source polygons use encoded pixels, irrespective of JPEG EXIF rotation."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    cv2.setNumThreads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--candidate", type=Path, default=Path("data/annotations/v5/candidates/totaltext_20261022/totaltext_train.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("data/annotations/v5/detection/totaltext_val.jsonl"))
    parser.add_argument("--train", type=Path, default=Path("data/annotations/v5/detection/totaltext_train.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("data/annotations/v5/detection/totaltext_test.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/annotations/v5/fusion"))
    parser.add_argument("--images", type=Path)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    bright_path = args.out / f"totaltext_{args.split}_bright.jsonl"
    dark_path = args.out / f"totaltext_{args.split}_dark.jsonl"
    protocol_path = args.out / f"totaltext_{args.split}_protocol.json"
    if any(path.exists() for path in (bright_path, dark_path, protocol_path)):
        raise FileExistsError("Preserving the existing fixed fusion development protocol")
    image_dir = args.root / (args.images or Path(f"derived/v5/fusion/totaltext_{args.split}_dark"))
    if image_dir.exists() and any(image_dir.iterdir()):
        raise FileExistsError(f"Preserving existing derived images: {image_dir}")

    if args.split == 'test':
        from tools.convert_annotations import totaltext_rows, STATS
        STATS.clear()
        candidate = totaltext_rows(str(args.root / 'raw'), 'test')
        exclusions = [dict(image=row['image'], **entry) for row in candidate
                      for entry in row['meta'].get('annotation_exclusions', [])]
        if (any(STATS[key] for key in ('tt_missing_image_skipped', 'tt_line_unparsed')) or
                STATS['tt_instance_skipped'] != len(exclusions) or
                any(not entry['ignore'] or len(entry['poly']) >= 3 for entry in exclusions)):
            raise ValueError(f'Incomplete released test annotation parsing: {dict(STATS)}')
    else:
        candidate = rows(args.candidate)
        exclusions = []
    by_id = {Path(row["image"]).stem: row for row in candidate}
    reference = rows(args.dev if args.split == 'dev' else args.test)
    ids = [Path(row["image"]).stem for row in reference]
    expected = 100 if args.split == 'dev' else 300
    if len(by_id) != len(candidate) or len(ids) != expected or len(set(ids)) != expected:
        raise ValueError(f"Expected unique corrected candidates and exactly {expected} fixed {args.split} IDs")
    selected = [copy.deepcopy(by_id[identifier]) for identifier in ids]
    if any(source["image"] != fixed["image"] for source, fixed in zip(selected, reference)):
        raise ValueError("A corrected image ID does not match its fixed development image path")

    fingerprints = {}
    def image_hash(relative):
        if relative not in fingerprints:
            fingerprints[relative] = sha256(args.root / relative)
        return fingerprints[relative]

    train_hashes = {image_hash(row["image"]) for row in rows(args.train)}
    test_hashes = {image_hash(row["image"]) for row in rows(args.test)}
    bright_hashes = {image_hash(row["image"]) for row in selected}
    dev_hashes = {image_hash(row['image']) for row in rows(args.dev)}
    overlaps = dict(train_test=len(train_hashes & test_hashes), dev_train=len(dev_hashes & train_hashes),
                    dev_test=len(dev_hashes & test_hashes))
    if any(overlaps.values()):
        raise ValueError(f"TotalText image-byte overlap: {overlaps}")
    expected_hashes = dev_hashes if args.split == 'dev' else test_hashes
    if bright_hashes != expected_hashes:
        raise ValueError('Selected image bytes differ from the registered split')

    args.out.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    bright_rows, dark_rows, pairs = [], [], []
    dark_hashes = set()
    for index, original in enumerate(selected):
        path = args.root / original["image"]
        image = encoded_rgb(path)
        rng_seed = (args.seed, index, list(PHYS_LEVELS).index("dark"))
        dark, metadata = darken(image, np.random.default_rng(rng_seed), level="dark", mode="physical")
        if dark.shape != image.shape:
            raise ValueError("Darkening changed image geometry")
        target = image_dir / f"{ids[index]}.png"
        if not cv2.imwrite(str(target), cv2.cvtColor(dark, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Could not write {target}")
        dark_hash = sha256(target)
        dark_hashes.add(dark_hash)
        held_other = test_hashes if args.split == 'dev' else dev_hashes
        if dark_hash in train_hashes or dark_hash in held_other:
            raise ValueError("A derived dark image overlaps training or another held-out split")
        bright, dark_row = copy.deepcopy(original), copy.deepcopy(original)
        bright["source_id"] = dark_row["source_id"] = original["image"]
        bright["lit"] = original["image"]
        bright.setdefault("meta", {}).update(split=args.split, fusion_pair_id=ids[index], illum_kind="lit",
                                              illum_y=luminance(image.astype(np.float32) / 255))
        bright["meta"]["illum_bin"] = illum_bin(bright["meta"]["illum_y"])
        dark_row["image"] = str(target.relative_to(args.root))
        dark_row["lit"] = original["image"]
        dark_row.setdefault("meta", {}).update(metadata, split=args.split, fusion_pair_id=ids[index],
                                               illum_kind="synth", rng_seed=list(rng_seed))
        bright_rows.append(bright)
        dark_rows.append(dark_row)
        pairs.append(dict(id=ids[index], bright=original["image"], dark=dark_row["image"],
                          width=image.shape[1], height=image.shape[0],
                          bright_sha256=image_hash(original["image"]), dark_sha256=dark_hash,
                          instances=len(original["instances"]),
                          ignored=sum(bool(instance["ignore"]) for instance in original["instances"])))

    for path, values in ((bright_path, bright_rows), (dark_path, dark_rows)):
        with path.open("x", encoding="utf-8") as stream:
            for row in values:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    conversion = args.candidate.parent / "conversion.json" if args.split == 'dev' else None
    source_annotations = (dict(path=str(args.candidate), sha256=sha256(args.candidate)) if args.split == 'dev' else
        {str(p): sha256(p) for p in sorted((args.root/'raw/TotalText/txt_format/Test').glob('poly_gt_*.txt'))})
    protocol = dict(role=("fixed paired fusion development set; not an official paper benchmark or final test" if args.split == 'dev'
                          else "fixed paired official TotalText test images; human GT; internal spotting protocol"),
                    split=args.split, annotation_source="human_dataset_release", source_annotation_sha256=source_annotations,
                    annotation_exclusions=exclusions,
                    exif_orientation='ignored; source polygons and synthesis share the encoded pixel frame',
                    samples=len(ids), image_ids=ids, seed=args.seed, mode="physical", level="dark",
                    rng=f"numpy.default_rng((seed, fixed_{args.split}_row_index, PHYS_LEVELS dark index=1))",
                    annotations=("All instances copied from corrected candidate without additional filtering; identical polygons/text/ignore in bright and dark"
                                 if args.split == 'dev' else "Released Test annotations reparsed; only explicitly recorded zero-area don't-care excluded; identical GT in bright and dark"),
                    pairing=f"source_id and lit are the original fixed-{args.split} image path in both arms; meta.fusion_pair_id is its image stem",
                    candidate_conversion=dict(path=str(conversion), sha256=sha256(conversion)) if conversion and conversion.exists() else None,
                    source_manifests={name: dict(path=str(path), sha256=sha256(path)) for name, path in
                                      ([('corrected_candidate', args.candidate)] if args.split == 'dev' else []) +
                                      [("fixed_dev_ids", args.dev), ("isolated_train", args.train),
                                       ("official_test_hash_check_only" if args.split == 'dev' else "official_test_ids", args.test)]},
                    outputs={name: dict(path=str(path), sha256=sha256(path)) for name, path in
                             (("bright", bright_path), ("dark", dark_path))},
                    image_byte_overlap={**overlaps, "dark_train": len(dark_hashes & train_hashes),
                                        "dark_dev": len(dark_hashes & dev_hashes),
                                        "dark_test": len(dark_hashes & test_hashes)},
                    synthesis_source=dict(path="seeread/data/lowlight_synth.py", sha256=sha256("seeread/data/lowlight_synth.py")),
                    preparation_source=dict(path=str(Path(__file__).relative_to(Path.cwd())), sha256=sha256(__file__)),
                    runtime=dict(numpy=np.__version__, opencv=cv2.__version__),
                    total_instances=sum(pair["instances"] for pair in pairs),
                    ignored_instances=sum(pair["ignored"] for pair in pairs), pairs=pairs)
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: protocol[key] for key in ("samples", "total_instances", "ignored_instances", "image_byte_overlap", "outputs")}, indent=2))


if __name__ == "__main__":
    main()
