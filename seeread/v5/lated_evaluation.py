"""LATeD development metrics with CTW-VOC matching and separate OCR eligibility.

Port of the matching rule in Yuliang-Liu/Curve-Text-Detector, revision
2126a660d3baf3e84a76e77352597576519c35a2 (MIT). Variable-length LATeD polygons
are consumed unchanged. This is a protocol port, not an author LATeD evaluator.
"""
from __future__ import annotations

from collections import Counter
import numpy as np
from shapely.geometry import Polygon

from seeread.metrics.spotting import edit_distance, norm_text
from seeread.v5.reading import metrics


def image_counts(predictions, ground_truth):
    """Input order, best-overall GT, strict IoU>0.5; duplicate match is FP.

    Unlike the old development evaluator, matching does not seek a second-best
    unused GT. Predictions matched to non-Latin/unreadable GT are detection
    predictions but are excluded from the Latin OCR precision denominator.
    Unmatched predictions remain OCR false positives.
    """
    polygons = [Polygon(g['poly']) for g in ground_truth]
    used, matches = set(), []
    counts = Counter(n_gt=sum(not g.get('ignore', False) for g in ground_truth),
                     n_gt_ocr=sum(bool(g.get('ocr_eligible')) and not g.get('ignore', False) for g in ground_truth))
    for index, prediction in enumerate(predictions):
        polygon = Polygon(prediction['poly'])
        if not polygon.is_valid or polygon.area <= 0:
            # The public evaluator skips invalid predicted polygons; expose count.
            counts['invalid_predictions'] += 1
            continue
        overlaps = []
        for other in polygons:
            if not other.is_valid or other.area <= 0:
                overlaps.append(0.)
                continue
            intersection = polygon.intersection(other).area
            overlaps.append(intersection / max(polygon.area + other.area - intersection, .00001))
        best = int(np.argmax(overlaps)) if overlaps else -1
        matched = best >= 0 and overlaps[best] > .5
        gt = ground_truth[best] if matched else None
        if matched and gt.get('ignore', False):
            continue
        counts['n_pred'] += 1
        eligible = matched and bool(gt.get('ocr_eligible'))
        if not matched or eligible:
            counts['n_pred_ocr'] += 1
        if matched and best not in used:
            used.add(best)
            counts['tp_det'] += 1
            matches.append((index, best))
            if eligible:
                reference = norm_text(gt['text'])
                counts['tp_e2e'] += bool(reference) and not prediction.get('crop_error') and norm_text(prediction['text']) == reference
    return dict(counts), matches


def summary(counts):
    def prf(tp, predicted, reference):
        p, r = tp / max(predicted, 1), tp / max(reference, 1)
        return dict(P=p, R=r, F1=2 * p * r / (p + r) if p + r else 0.)
    return dict(counts=dict(counts), detection=prf(counts.get('tp_det', 0), counts.get('n_pred', 0), counts.get('n_gt', 0)),
                latin_e2e=prf(counts.get('tp_e2e', 0), counts.get('n_pred_ocr', 0), counts.get('n_gt_ocr', 0)))


def evaluate_records(rows, records):
    if [r['image'] for r in rows] != [r['image'] for r in records]:
        raise ValueError('Evaluation requires every source image exactly once in source order')
    total, per_image = Counter(), []
    gt_words, matched_gt, matched_pred = [], [], []
    for row, record in zip(rows, records):
        counts, matches = image_counts(record['predictions'], row['instances'])
        total.update(counts)
        crop_list = record.get('gt_crops', [])
        crops = {c['id']: c for c in crop_list}
        expected = {g['id'] for g in row['instances'] if g.get('ocr_eligible') and not g.get('ignore')}
        if len(crops) != len(crop_list) or set(crops) != expected:
            raise ValueError('GT crop records must include all eligible lines, including failed crops')
        current = []
        for gt in row['instances']:
            if gt['id'] in expected:
                item = dict(text=gt['text'], prediction=crops[gt['id']]['text'])
                current.append(item); gt_words.append(item)
        for pi, gi in matches:
            gt = row['instances'][gi]
            if gt['id'] in expected:
                matched_gt.append(dict(text=gt['text'], prediction=crops[gt['id']]['text']))
                matched_pred.append(dict(text=gt['text'], prediction=record['predictions'][pi]['text']))
        raw = metrics(current)
        per_image.append(dict(image=row['image'], **counts, edits=raw['edit_operations'], chars=raw['reference_characters']))
    return dict(**summary(total), gt_crop_raw=metrics(gt_words), matched_gt_crop_raw=metrics(matched_gt),
                matched_prediction_crop_raw=metrics(matched_pred), per_image=per_image)


def paired_comparison(reference, candidate, iterations=2000):
    a, b = reference['per_image'], candidate['per_image']
    if [r['image'] for r in a] != [r['image'] for r in b] or not a:
        raise ValueError('Paired bootstrap needs the same nonempty source image list')
    keys = ('tp_e2e', 'n_pred_ocr', 'n_gt_ocr', 'edits', 'chars')
    arrays = [np.array([[row.get(k, 0) for k in keys] for row in rows], dtype=float) for rows in (a, b)]
    def score(values):
        tp, pred, gt, edits, chars = values.T
        return np.stack([2 * tp / np.maximum(pred + gt, 1), edits / np.maximum(chars, 1)], -1)
    observed = score(arrays[1].sum(0)[None]) - score(arrays[0].sum(0)[None])
    indices = np.random.default_rng(0).integers(0, len(a), (iterations, len(a)))
    delta = score(arrays[1][indices].sum(1)) - score(arrays[0][indices].sum(1))
    intervals = np.quantile(delta, [.025, .975], axis=0)
    return dict(e2e_F1_delta=float(observed[0, 0]), gt_CER_delta=float(observed[0, 1]),
                e2e_F1_95ci=intervals[:, 0].tolist(), gt_CER_95ci=intervals[:, 1].tolist(),
                image_bootstrap_iterations=iterations, seed=0, gain_gate=bool(observed[0, 0] >= .03))


def reliable_training_matches(predictions, ground_truth):
    """High-purity one-to-one training matches; never use recognized text."""
    gt = [Polygon(g['poly']) for g in ground_truth]
    candidates = []
    for pi, prediction in enumerate(predictions):
        polygon = Polygon(prediction['poly'])
        if not polygon.is_valid or polygon.area <= 0:
            continue
        overlaps = [polygon.intersection(p).area if p.is_valid and p.area > 0 else 0. for p in gt]
        if sum(a / max(p.area, 1e-9) >= .5 for a, p in zip(overlaps, gt)) > 1:
            continue
        for gi, (area, target) in enumerate(zip(overlaps, gt)):
            if not ground_truth[gi].get('ocr_eligible') or ground_truth[gi].get('ignore'):
                continue
            score = area / max(polygon.area + target.area - area, 1e-9)
            if score >= .7 and area / polygon.area >= .9 and area / max(target.area, 1e-9) >= .9:
                candidates.append((score, prediction.get('score', 0), pi, gi))
    used_pred, used_gt, matches = set(), set(), []
    for _, _, pi, gi in sorted(candidates, key=lambda v: (-v[0], -v[1], v[2], v[3])):
        if pi not in used_pred and gi not in used_gt:
            matches.append((pi, gi)); used_pred.add(pi); used_gt.add(gi)
    return matches
