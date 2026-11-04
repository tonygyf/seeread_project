"""GT-crop STR preparation; no detector, label generation or test-set tuning."""
from collections import Counter
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch

from seeread.v5.reading import metrics, sha256
from seeread.v5.spotting import polygon_crop


def audit_splits(root, splits):
    seen_images, seen_hashes = {}, {}
    result = {}
    for split, rows in splits.items():
        instances = 0
        for row in rows:
            image = row['image']
            if image in seen_images:
                raise ValueError(f'Duplicate source image: {image}')
            seen_images[image] = split
            digest = sha256(Path(root) / image)
            if digest in seen_hashes and seen_hashes[digest] != split:
                raise ValueError(f'Image content crosses splits: {image}')
            seen_hashes[digest] = split
            instances += len(row['instances'])
        result[split] = dict(images=len(rows), instances=instances,
                            usable_transcriptions=sum(bool(g.get('text')) and g['text'] != '###'
                                                      for r in rows for g in r['instances'] if not g.get('ignore')))
    return result


def quality_hint(rgb):
    """Flag missing pixel information; never infer text readability from OCR."""
    channel_ranges = np.ptp(rgb.reshape(-1, 3).astype(np.int16), axis=0)
    return dict(needs_extreme_review=bool(channel_ranges.max() <= 1),
                reason='near_constant_pixels' if channel_ranges.max() <= 1 else None,
                decision='human_review_required', automatic_exclusion=False)


def crop_views(rgb, polygon, method='axis_aligned'):
    """Keep native AABB for quality review; optionally use the deployed rectifier."""
    if method not in ('axis_aligned', 'boundary_when_supported'):
        raise ValueError('Unknown crop method')
    poly = np.asarray(polygon, dtype=np.float32)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3 or not np.isfinite(poly).all():
        return None, {}, 'invalid_polygon'
    h, w = rgb.shape[:2]
    if np.any(poly < 0) or np.any(poly[:, 0] > w) or np.any(poly[:, 1] > h):
        return None, {}, 'polygon_outside_image'
    lo = np.floor(poly.min(0)).astype(int)
    hi = np.minimum(np.ceil(poly.max(0)).astype(int) + 1, [w, h])
    native = rgb[lo[1]:hi[1], lo[0]:hi[0]]
    if min(native.shape[:2]) < 2 or abs(cv2.contourArea(poly)) < 1:
        return None, {}, 'degenerate_polygon'
    gray = cv2.cvtColor(native, cv2.COLOR_RGB2GRAY)
    quality = dict(native_hw=list(native.shape[:2]), gray_mean=float(gray.mean()),
                   gray_std=float(gray.std()), laplacian_variance=float(cv2.Laplacian(gray, cv2.CV_32F).var()),
                   polygon_vertices=len(poly), quality_is_readability_label=False)
    if method == 'axis_aligned':
        return native, quality, None
    result = polygon_crop(torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1),
                          poly, output_hw=(384, 384), sampling=method, pad=1.1)
    quality.update(sampling=result.sampling, fallback_reason=result.fallback_reason)
    if result.crop is None:
        return None, quality, result.reason
    return result.crop[0].permute(1, 2, 0).mul(255).round().byte().numpy(), quality, None


def export_reviewed(rows, root, image_dir, split, method):
    """Input comes only from validated human sidecars; keep an all-instance ledger."""
    root, image_dir = Path(root).resolve(), Path(image_dir).resolve()
    image_dir.relative_to(root)
    words, catalog = [], []
    for row in rows:
        image_path = root / row['image']
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if bgr is None:
            raise FileNotFoundError(image_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for gt in row['instances']:
            record = dict(id=gt['id'], source_image=row['image'], split=split, poly=gt['poly'],
                          text=gt['text'], language=gt.get('lang'),
                          transcription_status=gt.get('transcription_status'), source_ignore=bool(gt.get('ignore', False)),
                          eligible=bool(gt.get('ocr_eligible')) and not gt.get('ignore', False),
                          crop_error=None, method=method)
            if record['eligible']:
                if record['transcription_status'] != 'reviewed' or not record['text'] or record['text'] == '###':
                    raise ValueError('A machine draft or missing text cannot become supervised STR truth')
                crop, quality, error = crop_views(rgb, gt['poly'], method)
                record.update(quality=quality, crop_error=error)
                if crop is not None:
                    name = hashlib.sha256(gt['id'].encode()).hexdigest()[:24] + '.png'
                    path = image_dir / split / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists():
                        raise FileExistsError(path)
                    if not cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)):
                        raise OSError(path)
                    word = dict(id=gt['id'], image=str(path.relative_to(root)), text=gt['text'],
                                source='LATeD_human_reviewed_GT', source_group=row['image'], split=split,
                                dataset='LATeD_scene_STR', scene_group_verified=False,
                                sha256={'image': sha256(path)})
                    words.append(word); record['crop_image'] = word['image']
            catalog.append(record)
    return words, catalog


def score_catalog(catalog, predictions):
    """Every readable reference counts; failed crops receive empty predictions."""
    ids = [r['id'] for r in predictions]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate prediction id')
    by_id = {r['id']: r for r in predictions}
    eligible = [r for r in catalog if r['eligible']]
    expected = {r['id'] for r in eligible if not r['crop_error']}
    if set(by_id) != expected:
        raise ValueError('Predictions must cover every successful eligible crop exactly')
    records = [dict(text=r['text'], prediction='' if r['crop_error'] else by_id[r['id']]['prediction'])
               for r in eligible]
    ignored = sum(bool(r.get('source_ignore')) for r in catalog)
    return dict(raw=metrics(records), total_source_instances=len(catalog), source_ignored=ignored,
                eligible_instances=len(eligible),
                eligible_fraction_of_nonignored=len(eligible) / (len(catalog)-ignored) if len(catalog)>ignored else None,
                unreadable_instances=sum(r.get('transcription_status') == 'unreadable' for r in catalog),
                reviewed_non_latin=sum(r.get('transcription_status') == 'reviewed' and r.get('language') != 'latin' for r in catalog),
                crop_failures=sum(bool(r['crop_error']) for r in eligible),
                status_counts=dict(Counter(r.get('transcription_status', 'unknown') for r in catalog)),
                scope='GT-crop STR, not predicted-box recognition or end-to-end spotting')


def pair_training(a, b):
    """Use common training IDs so a crop ablation does not also change labels."""
    aa, bb = {r['id']: r for r in a}, {r['id']: r for r in b}
    if len(aa) != len(a) or len(bb) != len(b):
        raise ValueError('Duplicate training id')
    common = aa.keys() & bb.keys()
    if not common:
        raise ValueError('No common training IDs')
    for key in common:
        if aa[key]['split'] != 'train' or bb[key]['split'] != 'train':
            raise ValueError('Pairing is training-only')
        if any(aa[key][field] != bb[key][field] for field in ('text', 'source_group', 'source')):
            raise ValueError('Training labels or source scenes differ')
    ids = sorted(common)
    return [aa[k] for k in ids], [bb[k] for k in ids]
