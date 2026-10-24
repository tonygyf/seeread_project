"""Paired development-only enhancement placement ablation; no training or test inference.

A/B share raw-image detections. All cases share one recognizer, enhancer and
detector checkpoint. C enhances only the detector input; its crops use raw RGB.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from seeread.config import load_config
from seeread.metrics.spotting import SpottingEvaluator
from seeread.v5.enhancers import SelfDACEEnhancer
from seeread.v5.reading import metrics, sha256
from seeread.v5.recognition import ReadingPipeline, build_pipeline
from seeread.v5.seeing import build_detector
from seeread.v5.seeing_decode import decode
from seeread.v5.spotting import V5Spotter
from tools.v5_reading import FORMAT as READING_FORMAT, plain, write_json, source_hashes as reading_sources
from tools.v5_train_detection import FORMAT as DETECTION_FORMAT, topology_signature, source_hashes as detection_sources

CASES = {'A': 'none', 'B': 'after_crop', 'C': 'before_detector'}


def association_diagnostics(predictions, ground_truth):
    """Development attribution only; never feeds GT back into grouping/cropping.

    Every nonzero intersection is recorded. Candidate merges cover >=50% of
    two GT instances; candidate fragmentation has >=2 predictions each covering
    >=10% of a GT, with >=50% of each prediction inside that GT. These counts
    describe geometry, not a proof of linguistic word/line identity.
    """
    gt = [(i, Polygon(item['poly'])) for i, item in enumerate(ground_truth) if not item.get('ignore')]
    gt = [(i, poly) for i, poly in gt if poly.is_valid and poly.area > 0]
    links, merged, fragments = [], [], {i: [] for i, _ in gt}
    for index, item in enumerate(predictions):
        poly = Polygon(item['poly'])
        if not poly.is_valid or poly.area <= 0:
            continue
        covered = []
        for gt_index, target in gt:
            overlap = poly.intersection(target).area
            if overlap <= 0:
                continue
            coverage, purity = overlap / target.area, overlap / poly.area
            links.append(dict(prediction_index=index, gt_index=gt_index,
                              gt_coverage=coverage, prediction_purity=purity,
                              iou=overlap / (poly.area + target.area - overlap)))
            if coverage >= .5:
                covered.append(gt_index)
            if coverage >= .1 and purity >= .5:
                fragments[gt_index].append(index)
        if len(covered) > 1:
            merged.append(index)
    return dict(links=links, merged_prediction_candidates=merged,
                fragmented_gt_candidates=[i for i, indices in fragments.items() if len(indices) > 1])


def check_inputs(config):
    if config.get('cases') != CASES:
        raise ValueError('This paired ablation requires exactly A=none, B=after_crop, C=before_detector')
    inputs = {**config['manifests'], 'protocol': config['protocol'], **config['weights']}
    missing = {key: str(path) if path else 'unset' for key, path in inputs.items()
               if not path or not Path(path).is_file()}
    return dict(ready=not missing, missing=missing,
                scope='development only; checkpoints shared across all three cases')


def paired_rows(manifests):
    def read(path):
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    bright, dark = read(manifests['bright']), read(manifests['dark'])
    if not bright or len(bright) != len(dark):
        raise ValueError('Bright/dark manifests must be nonempty and paired')
    ids = set()
    for light, low in zip(bright, dark):
        source = light['source_id']
        if source != low['source_id'] or source in ids:
            raise ValueError('Bright/dark source IDs must match uniquely in manifest order')
        ids.add(source)
        if light['instances'] != low['instances']:
            raise ValueError('Paired images must retain identical original-coordinate annotations')
        for row in (light, low):
            if row.get('split', row.get('meta', {}).get('split', 'dev')) not in ('dev', 'val'):
                raise ValueError('Only development rows may enter this ablation')
            if any(part.lower() == 'test' for part in Path(source).parts):
                raise ValueError('Official test images cannot enter development ablations')
    return dict(bright=bright, dark=dark)


def verified_rows(config):
    rows = paired_rows(config['manifests'])
    protocol = json.loads(Path(config['protocol']).read_text())
    for illumination, samples in rows.items():
        if sha256(config['manifests'][illumination]) != protocol['outputs'][illumination]['sha256']:
            raise ValueError('Manifest SHA256 differs from the fixed development protocol')
        if len(samples) != protocol['samples'] or len(samples) != len(protocol['pairs']):
            raise ValueError('Development pair count differs from the fixed protocol')
        for row, pair in zip(samples, protocol['pairs']):
            if row['image'] != pair[illumination] or row['source_id'] != pair['bright']:
                raise ValueError('Development image order/identity differs from the fixed protocol')
            if sha256(Path(config['data_root']) / row['image']) != pair[illumination + '_sha256']:
                raise ValueError('Development image bytes differ from the fixed protocol')
    return rows


@torch.no_grad()
def evaluate_cases(image, ground_truth, detector, recognizer, enhancer, *, detector_hw, decoder, crop):
    """One image, three placements; recognition runs against raw-image polygons."""
    plain_reading = ReadingPipeline(nn.Identity(), recognizer).eval()
    enhanced_reading = ReadingPipeline(enhancer, recognizer).eval()
    plain_spotter = V5Spotter(detector, plain_reading, detector_hw=detector_hw,
                             decode_kwargs=decoder, crop_kwargs=crop).eval()
    enhanced_spotter = V5Spotter(detector, enhanced_reading, detector_hw=detector_hw,
                                decode_kwargs=decoder, crop_kwargs=crop).eval()
    canvas = F.interpolate(image[None], size=tuple(detector_hw), mode='bilinear', align_corners=False)
    raw_polygons = decode(detector(canvas), image_hw=tuple(image.shape[-2:]), **decoder)
    enhanced_canvas = enhancer(canvas)
    if enhanced_canvas.shape != canvas.shape or not torch.isfinite(enhanced_canvas).all():
        raise ValueError('Full-image enhancer must preserve RGB dimensions and produce finite output')
    enhanced_polygons = decode(detector(enhanced_canvas), image_hw=tuple(image.shape[-2:]), **decoder)

    def recognize(spotter, detections):
        predictions = spotter.recognize_polygons(image, [item['poly'] for item in detections])
        for prediction, detection in zip(predictions, detections):
            prediction['score'] = float(detection['score'])
        return predictions

    eligible_gt = [gt for gt in ground_truth if not gt.get('ignore') and gt.get('text', '') not in ('', '###')]

    def gt_crops(spotter):
        predictions = spotter.recognize_polygons(image, [gt['poly'] for gt in eligible_gt])
        # Failed geometric crops stay in the CER denominator with empty text.
        return [dict(text=gt['text'], prediction=prediction['text'], crop_error=prediction['crop_error'])
                for gt, prediction in zip(eligible_gt, predictions)]

    gt_plain, gt_enhanced = gt_crops(plain_spotter), gt_crops(enhanced_spotter)
    return {'A': dict(predictions=recognize(plain_spotter, raw_polygons), gt_crops=gt_plain),
            'B': dict(predictions=recognize(enhanced_spotter, raw_polygons), gt_crops=gt_enhanced),
            'C': dict(predictions=recognize(plain_spotter, enhanced_polygons), gt_crops=gt_plain)}


def load_models(weights, device, *, allow_pretraining=False):
    detection = torch.load(weights['detector'], map_location='cpu', weights_only=False)
    reading = torch.load(weights['reading'], map_location='cpu', weights_only=False)
    for saved, expected in ((detection, DETECTION_FORMAT), (reading, READING_FORMAT)):
        if saved.get('format') != expected or saved.get('resume_safe') is not True:
            raise ValueError('Only healthy, independent V5 checkpoints are accepted')
    if detection['config']['stage']['name'] != 'target' and not allow_pretraining:
        raise ValueError('Use a development-selected target detector, not an intermediate pretraining stage')
    detector = build_detector(detection['config']['detector'])
    if topology_signature(detector) != detection['topology']:
        raise ValueError('Detection implementation differs from the selected checkpoint')
    detector.load_state_dict(detection['model'], strict=True)
    pipeline = build_pipeline(reading['config'])
    pipeline.load_state_dict(reading['model'], strict=True)
    if not isinstance(pipeline.enhancer, SelfDACEEnhancer):
        raise ValueError('The placement ablation needs one genuine Self-DACE + TrOCR checkpoint')
    detector.requires_grad_(False).eval().to(device)
    pipeline.requires_grad_(False).eval().to(device)
    provenance = dict(detector=detection.get('provenance'), reading=reading.get('provenance'),
                      detector_step=detection['state']['step'], reading_step=reading['state']['global_step'])
    return detector, pipeline.recognizer, pipeline.enhancer, provenance


def rgb_tensor(path, device):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise FileNotFoundError(f'Cannot decode RGB image: {path}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().to(device) / 255


def run(config, device='cuda:0'):
    check = check_inputs(config)
    if not check['ready']:
        raise FileNotFoundError(json.dumps(check['missing'], ensure_ascii=False))
    rows = verified_rows(config)
    if device != 'cuda:0' or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Run in an authorized single-GPU container using logical cuda:0')
    torch.cuda.set_per_process_memory_fraction(float(config.get('memory_fraction', .35)), 0)
    out = Path(config['output'])
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status='initializing', config=config, cases=CASES,
                  weight_sha256={key: sha256(path) for key, path in config['weights'].items()},
                  manifest_sha256={key: sha256(path) for key, path in config['manifests'].items()},
                  protocol_sha256=sha256(config['protocol']),
                  source_sha256={**reading_sources(), **detection_sources(),
                                 **{str(path.relative_to(ROOT)): sha256(path) for path in
                                    (Path(__file__).resolve(), ROOT / 'seeread/v5/spotting.py')}},
                  metric_scope='internal development score-greedy IoU>=0.5, lower_alnum E2E; not official test',
                  gt_cer='all nonignored transcribed GT, including empty predictions for invalid geometric crops',
                  geometry='same frozen crop/decode settings for all cases; C crops original RGB',
                  gt_cer_C='same recognition graph as A; GT-crop predictions reused')
    write_json(out / 'run.json', report)
    try:
        detector, recognizer, enhancer, provenance = load_models(config['weights'], device)
        report.update(status='running', provenance=provenance, results={})
        write_json(out / 'run.json', report)
        for illumination, samples in rows.items():
            evaluators = {case: SpottingEvaluator(iou_thr=.5) for case in CASES}
            words = {case: [] for case in CASES}
            failures = {case: Counter() for case in CASES}
            associations = {case: Counter() for case in CASES}
            elapsed = 0.
            with open(out / (illumination + '.records.jsonl'), 'x') as stream:
                for row in samples:
                    image = rgb_tensor(Path(config['data_root']) / row['image'], device)
                    torch.cuda.synchronize()
                    started = time.monotonic()
                    cases = evaluate_cases(image, row['instances'], detector, recognizer, enhancer,
                                           detector_hw=config['detector_hw'], decoder=config['decoder'], crop=config['crop'])
                    torch.cuda.synchronize()
                    elapsed += time.monotonic() - started
                    for case, result in cases.items():
                        evaluators[case].add(result['predictions'], row['instances'])
                        words[case].extend(result['gt_crops'])
                        failures[case].update(p['crop_error'] for p in result['predictions'] if p['crop_error'])
                        predictions = [dict(item, poly=np.asarray(item['poly']).tolist()) for item in result['predictions']]
                        geometry = association_diagnostics(predictions, row['instances'])
                        associations[case].update({key: len(geometry[key]) for key in
                                                  ('merged_prediction_candidates', 'fragmented_gt_candidates')})
                        stream.write(json.dumps(dict(image=row['image'], source_id=row['source_id'], case=case,
                                                     predictions=predictions, gt_crops=result['gt_crops'],
                                                     association_diagnostics=geometry), allow_nan=False) + '\n')
            report['results'][illumination] = dict(
                complete=True, images=len(samples), all_case_seconds=elapsed,
                cases={case: dict(spotting=evaluators[case].summary()['all'],
                                  gt_crop_raw=metrics(words[case], 'raw'),
                                  gt_crop_lower_alnum=metrics(words[case], 'lower_alnum'),
                                  prediction_crop_failures=dict(failures[case]),
                                  association_candidates=dict(associations[case]),
                                  gt_crop_failures=sum(bool(word['crop_error']) for word in words[case]))
                       for case in CASES})
            write_json(out / 'run.json', report)
        report['status'] = 'complete'
        write_json(out / 'run.json', report)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(out / 'run.json', report)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--check-only', action='store_true', help='List missing inputs without loading models or accessing CUDA')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    config = plain(load_config(args.config))
    if args.check_only:
        result = check_inputs(config)
        if not ({'bright', 'dark', 'protocol'} & result['missing'].keys()):
            rows = verified_rows(config)
            result['verified_dev_images'] = {key: len(value) for key, value in rows.items()}
    else:
        result = run(config, args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
