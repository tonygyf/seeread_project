"""Verified-GT export for pinned author evaluators; does not run or replace them."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

TT_COMMIT = "59538b936fc7a55025987960270267d8d081e72e"
CTW_COMMIT = "2126a660d3baf3e84a76e77352597576519c35a2"
SOURCES = {
    "ctw1500_voc": dict(repo="https://github.com/Yuliang-Liu/Curve-Text-Detector", commit=CTW_COMMIT,
        files={"tools/ctw1500_evaluation/voc_eval_polygon.py": "c5ac67bd75d965f72c156ff65fc8e72d20ce0f936969dafb9f1e120fa1eb51e0",
               "tools/ctw1500_evaluation/test_ctw1500_eval.py": "b8aa266532e50504cfd7c584baedb780f5db79b5bcd3b4cbdccbccb5f54dd6f4"}),
    "totaltext_deteval": dict(repo="https://github.com/cs-chan/Total-Text-Dataset", commit=TT_COMMIT,
        files={"Evaluation_Protocol/Python_scripts/Deteval.py": "48cc45b7537c8850380a8397b15d7c57c01465c933a4678388b13b9be395b1b4",
               "Evaluation_Protocol/Python_scripts/polygon_fast.py": "352fa25026d899763725cfba3e2e7d022c6ab38d69e2b58401f7988c43959aa5"}),
    "totaltext_pascal": dict(repo="https://github.com/cs-chan/Total-Text-Dataset", commit=TT_COMMIT,
        files={"Evaluation_Protocol/Python_scripts/Pascal_VOC.py": "f6a4c7c16eedf25d35d6b9030677bbea89626a2287cc433909c748d64181de20",
               "Evaluation_Protocol/Python_scripts/polygon_fast.py": "352fa25026d899763725cfba3e2e7d022c6ab38d69e2b58401f7988c43959aa5"}),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _jsonl(path):
    with Path(path).open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError("Empty input manifest/predictions")
    result = {}
    stems = set()
    for row in rows:
        image = row["image"]
        if not isinstance(image, str) or Path(image).is_absolute() or ".." in Path(image).parts:
            raise ValueError("Image must be a safe data-root-relative path")
        stem = Path(image).stem
        if image in result or stem in stems:
            raise ValueError("Duplicate image or output stem naming collision")
        stems.add(stem)
        result[image] = row
    return result


def _polygon(value):
    points = [[float(x), float(y)] for x, y in value]
    if len(points) < 3 or not all(math.isfinite(v) for p in points for v in p):
        raise ValueError("Polygon must have at least three finite xy vertices")
    from shapely.geometry import Polygon
    q = Polygon(points)
    if not q.is_valid or q.area <= 0:
        raise ValueError("Invalid/zero-area polygon; exporter does not repair geometry")
    return points


def _same_polygon(a, b):
    # Source GT includes single-point don't-care and self-intersecting rings.
    # Compare their evidence exactly; prediction validity is checked by _polygon.
    a, b = ([[float(x), float(y)] for x, y in ring] for ring in (a, b))
    if not all(math.isfinite(v) for ring in (a, b) for point in ring for v in point):
        raise ValueError("Source GT coordinates must be finite")
    return len(a) == len(b) and any(a == ring[k:] + ring[:k]
                                  for ring in (b, b[::-1]) for k in range(len(b)))


def _check_instances(row, original):
    converted = row["instances"]
    if len(converted) != len(original):
        raise ValueError(f"Original GT instance count mismatch: {row['image']}")
    for index, (a, b) in enumerate(zip(converted, original)):
        if bool(a.get("ignore", False)) != b["ignore"] or not _same_polygon(a["poly"], b["poly"]):
            raise ValueError(f"Original GT polygon/ignore/order mismatch: {row['image']} instance {index}")


def _tt_gt(path):
    import numpy as np
    from scipy.io import loadmat
    table = loadmat(path)["polygt"]
    if table.ndim != 2 or table.shape[1] != 6:
        raise ValueError("Expected author Total-Text polygt Nx6 MATLAB table")
    original = []
    for gt in table:
        x, y = np.asarray(gt[1]).reshape(-1), np.asarray(gt[3]).reshape(-1)
        if len(x) != len(y):
            raise ValueError("GT coordinate lengths differ")
        marker = "".join(str(v) for v in np.asarray(gt[5]).reshape(-1))
        original.append(dict(poly=list(zip(x.tolist(), y.tolist())), ignore=marker == "#"))
    return original


def _ctw_gt(path):
    original = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        values = [int(v) for v in line.strip().split(",")]
        if len(values) != 32:
            raise ValueError("Expected original CTW bbox+28 offsets (32 integers)")
        original.append(dict(poly=[[values[0] + values[k], values[1] + values[k + 1]]
                                   for k in range(4, 32, 2)], ignore=False))
    return original


def export_predictions(manifest, predictions, output, *, protocols, data_root, gt_root,
                       ctw_imageset=None, ctw_annotations=None, quantization="none"):
    """Export complete original-coordinate records, after exact raw-GT validation.

    CTW list entries resolve relative to gt_root. Both lists must retain the author
    order (1001 -> evaluator id 0). TT requires original poly_gt_<stem>.mat files.
    TT integer quantization is explicit: truncate or nearest_even. No score filter,
    score sorting, polygon repair, ignore filtering or GT conversion is performed.
    """
    protocols = list(protocols)
    if not protocols or len(set(protocols)) != len(protocols) or any(p not in SOURCES for p in protocols):
        raise ValueError("Choose distinct explicit supported protocols")
    is_ctw = protocols == ["ctw1500_voc"]
    if "ctw1500_voc" in protocols and not is_ctw:
        raise ValueError("CTW and Total-Text require separate dataset exports")
    if is_ctw and quantization != "none":
        raise ValueError("CTW float export requires quantization=none")
    if quantization not in ("none", "truncate", "nearest_even") or (not is_ctw and quantization == "none"):
        raise ValueError("TT requires explicit integer quantization: truncate or nearest_even")
    rows, records = _jsonl(manifest), _jsonl(predictions)
    if rows.keys() != records.keys():
        raise ValueError("Predictions must cover every manifest image exactly, including empty predictions")
    output, data_root, gt_root = Path(output), Path(data_root), Path(gt_root)
    if output.exists():
        raise FileExistsError("Export output must not already exist")
    mappings, gt_files, list_hashes = {}, {}, {}
    if is_ctw:
        if ctw_imageset is None or ctw_annotations is None:
            raise ValueError("Original CTW image and annotation lists are required")
        images = Path(ctw_imageset).read_text().splitlines()
        annotations = Path(ctw_annotations).read_text().splitlines()
        if len(images) != len(rows) or len(annotations) != len(rows):
            raise ValueError("CTW original lists and manifest must have identical image counts")
        stems = {Path(image).stem: image for image in rows}
        for index, (name, annotation) in enumerate(zip(images, annotations)):
            stem = Path(name.strip()).stem
            if stem not in stems or not stem.isdigit() or int(stem) - 1001 != index or stem in mappings:
                raise ValueError("CTW original image order/id mapping conflicts with author sortdetection convention")
            path = gt_root / annotation.strip()
            if path.stem != stem:
                raise ValueError("CTW annotation/image naming mismatch")
            _check_instances(rows[stems[stem]], _ctw_gt(path))
            mappings[stem] = dict(image=stems[stem], evaluator_id=str(index), gt=str(path))
            gt_files[str(path)] = sha256(path)
        list_hashes = {str(p): sha256(p) for p in (ctw_imageset, ctw_annotations)}
    else:
        for image, row in rows.items():
            stem = Path(image).stem
            path = gt_root / ("poly_gt_" + stem + ".mat")
            if not path.is_file():
                raise FileNotFoundError(f"Original author MATLAB GT required (txt/MMOCR is insufficient): {path}")
            _check_instances(row, _tt_gt(path))
            mappings[stem] = dict(image=image, evaluator_id=stem, gt=str(path))
            gt_files[str(path)] = sha256(path)
    # Verify original pixel dimensions before any output directory exists.
    from PIL import Image
    canonical = []
    for image, row in rows.items():
        record = records[image]
        image_path = data_root / image
        with Image.open(image_path) as im:
            hw = [im.height, im.width]
        if record.get("original_hw") != hw:
            raise ValueError(f"Original image coordinate dimensions mismatch: {image}")
        image_hash = sha256(image_path)
        if record.get("image_sha256") is not None and record["image_sha256"] != image_hash:
            raise ValueError(f"Prediction image hash mismatch: {image}")
        items = []
        for item in record["predictions"]:
            score = float(item["score"])
            if not math.isfinite(score):
                raise ValueError("Nonfinite prediction score")
            items.append(dict(poly=_polygon(item["poly"]), score=score))
        canonical.append(dict(image=image, original_hw=hw, image_sha256=image_hash, predictions=items))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + output.name + ".", dir=output.parent))
    try:
        hashes, commands = {}, {}
        for protocol in protocols:
            directory = temporary / protocol
            directory.mkdir()
            merged = []
            ordered = (sorted(canonical, key=lambda r: int(mappings[Path(r["image"]).stem]["evaluator_id"]))
                       if is_ctw else canonical)
            for record in ordered:
                stem = Path(record["image"]).stem
                lines = []
                for item in record["predictions"]:
                    poly = item["poly"]
                    if is_ctw:
                        values = [format(v, ".17g") for p in poly for v in p]
                        merged.append(" ".join([mappings[stem]["evaluator_id"], "0.999", *values]))
                    else:
                        quantize = math.trunc if quantization == "truncate" else round
                        values = [str(quantize(v)) for x, y in poly for v in (y, x)]
                        # Quantization must not silently collapse/cross the polygon.
                        _polygon([[int(values[k + 1]), int(values[k])] for k in range(0, len(values), 2)])
                    lines.append(",".join(values))
                path = directory / (stem + ".txt")
                path.write_text("".join(line + "\n" for line in lines))
            if is_ctw:
                (temporary / "ctw1500_voc_detections_text0.5.txt").write_text("".join(s + "\n" for s in merged))
                commands[protocol] = dict(status="not_executed", instructions="Use pinned voc_eval_polygon(detpath, original_annotation_list, original_image_list, 'text', ovthresh=0.5) in an isolated audited author environment. The merged ../ctw1500_voc_detections_text0.5.txt is ready; do not pass it through sortdetection.py again.")
            else:
                script = "Deteval.py" if protocol == "totaltext_deteval" else "Pascal_VOC.py"
                commands[protocol] = dict(status="not_executed", instructions=f"In a separate pinned author source copy, set input_dir to {output / protocol}, gt_dir to the validated original MAT directory, and fid_path OUTSIDE the prediction directory, then run python {script}. Log the path-only patch and interpreter/dependency versions; do not use the modified local DarkText script.")
            for path in directory.iterdir():
                hashes[str(path.relative_to(temporary))] = sha256(path)
        if is_ctw:
            merged_path = temporary / "ctw1500_voc_detections_text0.5.txt"
            hashes[merged_path.name] = sha256(merged_path)
        path = temporary / "canonical_predictions.jsonl"
        path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in canonical))
        hashes[path.name] = sha256(path)
        audit = dict(schema="seeread_v5_author_eval_export_v1", evaluation_executed=False,
            paper_protocol_identity_confirmed=False, selection="explicit user protocol choice; no automatic comparison or selection",
            coordinate_frame="original encoded image pixels, no EXIF transform", quantization=quantization,
            prediction_order="input per-image order preserved; no score threshold/sort/NMS",
            gt_equivalence="instance order and ignore exact; only ring rotation/reversal permitted",
            protocols={p: SOURCES[p] for p in protocols}, source_verification="reference hashes recorded; no evaluator code executed or silently downloaded",
            manifest=dict(path=str(manifest), sha256=sha256(manifest), images=len(rows)),
            input_predictions=dict(path=str(predictions), sha256=sha256(predictions)),
            original_gt_sha256=gt_files, original_list_sha256=list_hashes, mapping=list(mappings.values()),
            output_sha256=hashes, external_evaluation=commands)
        (temporary / "export_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
        if output.exists():
            raise FileExistsError("Concurrent output appeared; refusing overwrite")
        os.rename(temporary, output)
        return audit
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
