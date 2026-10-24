"""Prepare pinned Reading word crops; verify existing archives before reuse.

No GPU or datasets library is needed. Images are streamed from ZIP members,
never loaded as a whole archive. Official test/test_real are registered only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import time
import urllib.parse
import urllib.request
import zipfile

REPO = "lumimusta/Low-light_Scene_Text_Dataset"
REVISION = "619e3c1d3bb27135da785b77d3c414a946e37000"
ARCHIVES = ("low_light_train.zip", "well_lit_train.zip", "low_light_test.zip",
            "well_lit_test.zip", "low_light_test_real.zip")
ANNOTATIONS = ("train.jsonl", "test.jsonl", "test_real.jsonl")


def digest_stream(stream, algorithm="sha256"):
    value = hashlib.new(algorithm)
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        value.update(chunk)
    return value.hexdigest()


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return digest_stream(stream)


def git_blob_sha1(path):
    value = hashlib.sha1(f"blob {Path(path).stat().st_size}\0".encode())
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def valid_hub_file(path, metadata):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != metadata["size"]:
        return False
    if metadata.get("lfs"):
        return file_sha256(path) == metadata["lfs"]["sha256"]
    return git_blob_sha1(path) == metadata["blobId"]


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive/annotation path: {name}")
    return path


def hub_metadata(revision):
    url = f"https://huggingface.co/api/datasets/{REPO}/revision/{revision}?blobs=true"
    with urllib.request.urlopen(url, timeout=90) as stream:
        result = json.load(stream)
    if result.get("sha") != revision:
        raise ValueError("Dataset revision is not an immutable matching commit")
    return result


def obtain_file(name, metadata, local, destination, revision, records):
    """Only reuse files authenticated against the pinned Hub blob/LFS digest."""
    safe_relative(name)
    candidate = local / name
    if valid_hub_file(candidate, metadata):
        records.append(dict(name=name, action="reuse_verified", path=str(candidate.resolve()),
                            size=candidate.stat().st_size, sha256=file_sha256(candidate),
                            hub_lfs_sha256=metadata.get("lfs", {}).get("sha256"),
                            hub_blob_id=metadata["blobId"]))
        return candidate
    target = destination / name
    if target.exists():
        if not valid_hub_file(target, metadata):
            raise ValueError(f"Existing v5 file has wrong official digest; preserved: {target}")
        action = "reuse_verified_v5"
    else:
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{urllib.parse.quote(name)}"
        temporary = target.with_name(target.name + ".part")
        if temporary.exists():
            raise FileExistsError(f"Previous partial download preserved: {temporary}")
        with urllib.request.urlopen(url, timeout=120) as source, temporary.open("xb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        if not valid_hub_file(temporary, metadata):
            raise ValueError(f"Downloaded file failed official digest: {temporary}")
        temporary.rename(target)
        action = "download_verified"
    records.append(dict(name=name, action=action, path=str(target.resolve()),
                        size=target.stat().st_size, sha256=file_sha256(target),
                        hub_lfs_sha256=metadata.get("lfs", {}).get("sha256"),
                        hub_blob_id=metadata["blobId"]))
    return target


def verified_members(archive, names, local, destination):
    """Compare each local image to its authenticated ZIP member's SHA256.

    A differing old image is preserved, and the authentic image is extracted
    only into the new v5 destination. Memory is bounded by the stream buffer.
    """
    result, stats = {}, Counter()
    with zipfile.ZipFile(archive) as zipped:
        for name in sorted(names):
            safe_relative(name)
            info = zipped.getinfo(name)
            if info.is_dir():
                raise ValueError(f"Image points at directory: {name}")
            with zipped.open(info) as stream:
                member_digest = digest_stream(stream)
            old = local / name
            if old.is_file() and old.stat().st_size == info.file_size and file_sha256(old) == member_digest:
                path = old
                stats["reused_verified_images"] += 1
            else:
                path = destination / name
                if path.exists():
                    if file_sha256(path) != member_digest:
                        raise ValueError(f"Existing v5 image differs; preserved: {path}")
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with zipped.open(info) as source, path.open("xb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                    if file_sha256(path) != member_digest:
                        raise ValueError(f"Extracted image failed digest: {path}")
                stats["extracted_images"] += 1
            result[name] = dict(path=path.resolve(), sha256=member_digest, size=info.file_size)
    return result, dict(stats)


def source_family(row, split):
    if split == "test_real":
        return "ESTR/real"
    if row.get("source"):
        return "LSTR/" + str(row["source"])
    identifier = str(row["id"])
    # These are filename strata, not a claim of verified original datasets.
    return "LSTR/" + ("word_prefix" if identifier.startswith("word_") else
                       "numeric_id" if identifier.isdecimal() else "underscore_or_other_id")


def attach_identity(rows):
    """Group duplicate light/dark crops together, including transitive pairs."""
    parents = {}

    def find(value):
        parents.setdefault(value, value)
        if parents[value] != value:
            parents[value] = find(parents[value])
        return parents[value]

    def union(left, right):
        a, b = find(left), find(right)
        if a != b:
            parents[max(a, b)] = min(a, b)

    for row in rows:
        hashes = row["sha256"]
        union(hashes["image"], hashes["lit"] or hashes["image"])
        # An official explicit scene mapping, if available, takes precedence.
        scene = row.get("verified_scene")
        if scene:
            union(hashes["image"], "scene:" + row["source"] + ":" + scene)
    for row in rows:
        row["source_group"] = "reading:" + find(row["sha256"]["image"])
        row["scene_group_verified"] = bool(row.pop("verified_scene", None))
    return rows


def split_manifests(rows, seed=0, dev_fraction=0.1):
    """Only split official train; stratify deterministic identity groups by source."""
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be in (0,1)")
    rows = attach_identity([dict(row, sha256=dict(row["sha256"])) for row in rows])
    official_train = [r for r in rows if r["dataset"]["original_split"] == "train"]
    heldout = [r for r in rows if r["dataset"]["original_split"] != "train"]
    heldout_groups = {r["source_group"] for r in heldout}
    quarantined = [r for r in official_train if r["source_group"] in heldout_groups]
    official_train = [r for r in official_train if r["source_group"] not in heldout_groups]
    groups = defaultdict(list)
    for row in official_train:
        groups[row["source_group"]].append(row)
    strata = defaultdict(list)
    for group, members in groups.items():
        strata["|".join(sorted({r["source"] for r in members}))].append(group)
    dev_groups = set()
    for stratum_groups in strata.values():
        ordered = sorted(stratum_groups, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
        n_dev = min(len(ordered) - 1, max(1, round(len(ordered) * dev_fraction))) if len(ordered) > 1 else 0
        dev_groups.update(ordered[:n_dev])
    output = {name: [] for name in ("train", "dev", "test", "test_real", "quarantine")}
    for row in official_train:
        split = "dev" if row["source_group"] in dev_groups else "train"
        output[split].append(dict(row, split=split))
    for row in heldout:
        split = row["dataset"]["original_split"]
        output[split].append(dict(row, split=split))
    for row in quarantined:
        output["quarantine"].append(dict(row, split="quarantine", exclusion_reason="content_group_overlaps_official_test"))
    for values in output.values():
        values.sort(key=lambda r: r["id"])
    return output


def validate_manifests(manifests, data_root):
    group_sets, path_sets, hash_sets = {}, {}, {}
    for split, rows in manifests.items():
        ids = set()
        group_sets[split], path_sets[split], hash_sets[split] = set(), set(), set()
        for row in rows:
            if row["id"] in ids:
                raise ValueError(f"Duplicate id in {split}: {row['id']}")
            ids.add(row["id"])
            group_sets[split].add(row["source_group"])
            for key in ("image", "lit"):
                if row[key] is None:
                    continue
                safe_relative(row[key])
                path = data_root / row[key]
                if not path.is_file():
                    raise FileNotFoundError(path)
                path_sets[split].add(str(path.resolve()))
                hash_sets[split].add(row["sha256"][key])
    # Official test/test_real are left intact even if their publisher reused content.
    for a, b in (("train", "dev"), ("train", "test"), ("train", "test_real"),
                 ("dev", "test"), ("dev", "test_real")):
        if group_sets[a] & group_sets[b] or path_sets[a] & path_sets[b] or hash_sets[a] & hash_sets[b]:
            raise ValueError(f"Split leakage between {a} and {b}")
    return dict(paths_exist=True, unique_ids_per_split=True, train_dev_groups_disjoint=True,
                training_and_official_test_content_disjoint=True,
                official_test_real_shared_content_hashes=len(hash_sets["test"] & hash_sets["test_real"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--local-dir", type=Path, default=Path("data/raw/ESTR"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/v5/reading"))
    parser.add_argument("--out", type=Path, default=Path("data/annotations/v5/reading"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    metadata = hub_metadata(args.revision)
    files = {f["rfilename"]: f for f in metadata["siblings"]}
    inventory = dict(repo=REPO, revision=args.revision,
                     official_bytes=sum(files[n]["size"] for n in ARCHIVES + ANNOTATIONS),
                     official_files={n: files[n] for n in ARCHIVES + ANNOTATIONS},
                     local_candidates={n: dict(exists=(args.local_dir / n).is_file(),
                                               size=(args.local_dir / n).stat().st_size if (args.local_dir / n).is_file() else None)
                                       for n in ARCHIVES + ANNOTATIONS})
    print(json.dumps(inventory, indent=2), flush=True)
    if args.inventory_only:
        return
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Preserving existing manifests; choose a new --out: {args.out}")
    started = time.monotonic()
    destination = args.raw_dir / args.revision
    destination.mkdir(parents=True, exist_ok=True)
    snapshot = destination / "hub_manifest.json"
    if not snapshot.exists():
        with snapshot.open("x") as stream:
            json.dump(metadata, stream, indent=2)
    downloads, metadata_paths = [], {}
    for name in ANNOTATIONS:
        path = obtain_file(name, files[name], args.local_dir, destination, args.revision, downloads)
        # Preserve a small pinned metadata copy even when the authenticated old copy is reused.
        target = destination / name
        if not target.exists():
            with path.open("rb") as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
        metadata_paths[name[:-6]] = target
    official_rows = {split: [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                     for split, path in metadata_paths.items()}
    print(json.dumps({"official_row_counts": {k: len(v) for k, v in official_rows.items()}}, indent=2), flush=True)
    expected_members = defaultdict(set)
    for rows in official_rows.values():
        for row in rows:
            for field in ("low_light", "well_lit"):
                if row.get(field):
                    name = row[field]
                    safe_relative(name)
                    expected_members[name.split("/")[0] + ".zip"].add(name)
    members, archive_stats = {}, {}
    for name in ARCHIVES:
        archive = obtain_file(name, files[name], args.local_dir, destination, args.revision, downloads)
        found, stats = verified_members(archive, expected_members[name], args.local_dir, destination)
        members.update(found); archive_stats[name] = stats
        print(json.dumps({"archive": name, **stats}), flush=True)
    rows = []
    for split, official in official_rows.items():
        for number, row in enumerate(official):
            low, lit = members[row["low_light"]], members.get(row.get("well_lit"))
            rows.append(dict(id=f"{split}:{row['id']}",
                             image=str(low["path"].relative_to(args.data_root.resolve())),
                             lit=str(lit["path"].relative_to(args.data_root.resolve())) if lit else None,
                             text=row["label"], source=source_family(row, split),
                             verified_scene=str(row.get("scene_id") or row.get("source_image") or "") or None,
                             dataset=dict(repo=REPO, revision=args.revision, config="default", original_split=split,
                                          row_id=row["id"], row_index=number),
                             sha256=dict(image=low["sha256"], lit=lit["sha256"] if lit else None)))
    manifests = split_manifests(rows, args.seed, args.dev_fraction)
    validation = validate_manifests(manifests, args.data_root.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_records = {}
    for split, values in manifests.items():
        path = args.out / f"{split}.jsonl"
        with path.open("x", encoding="utf-8") as stream:
            for row in values:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest_records[split] = dict(path=str(path.resolve()), rows=len(values), sha256=file_sha256(path),
                                       source_counts=dict(Counter(r["source"] for r in values)),
                                       groups=len({r["source_group"] for r in values}))
    provenance = dict(schema="reading_word_crop_v1", repo=REPO, revision=args.revision,
                      official_url=f"https://huggingface.co/datasets/{REPO}/tree/{args.revision}",
                      inventory=inventory, files=downloads, archives=archive_stats,
                      original_counts={k: len(v) for k, v in official_rows.items()}, manifests=manifest_records,
                      validation=validation, seed=args.seed, dev_fraction=args.dev_fraction,
                      policy=dict(unit="word crop; not scene spotting; no invented polygons",
                                  holdouts="official test and test_real registered only; never used for hyperparameter selection",
                                  split="Official train only; filename-source-stratified content groups; stable SHA256 order, seed0",
                                  groups="Union identical dark/lit byte hashes transitively; verified scene mapping only when explicit",
                                  scene_limit="Official metadata lacks source-scene mapping. Unknown scenes use crop content identity; scene_group_verified=false. No claim of complete scene isolation.",
                                  sources="Missing official source labels are filename-family strata, not authenticated dataset identities",
                                  content_overlap="Official train groups sharing any dark/lit hash with holdouts are quarantined before train/dev splitting",
                                  real="60 ESTR real word crops are recognition-only test data, not full spotting images"),
                      elapsed_seconds=time.monotonic() - started)
    with (args.out / "provenance.json").open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"manifests": manifest_records, "validation": validation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
