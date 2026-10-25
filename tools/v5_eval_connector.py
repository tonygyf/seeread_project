"""Fixed intermediate detector and Reading weights: crop-only development comparison."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from seeread.config import load_config
from seeread.metrics.spotting import SpottingEvaluator
from seeread.v5.reading import metrics, sha256
from seeread.v5.recognition import ReadingPipeline
from seeread.v5.seeing_decode import decode
from seeread.v5.spotting import V5Spotter
from tools.v5_detection_pilot import plain, require_cuda, write_json
from tools.v5_eval_spotting import verified_rows, rgb_tensor, load_models, association_diagnostics
from tools.v5_reading import source_hashes as reading_sources
from tools.v5_train_detection import source_hashes as detection_sources


@torch.inference_mode()
def compare_image(image, ground_truth, detector, reading, config, samplings=('local_normal', 'polygon_boundary')):
    canvas = F.interpolate(image[None], size=tuple(config['detector_hw']), mode='bilinear', align_corners=False)
    detections = decode(detector(canvas), image_hw=tuple(image.shape[-2:]), **config['decoder'])
    eligible = [(i, gt) for i, gt in enumerate(ground_truth)
                if not gt.get('ignore') and gt.get('text', '') not in ('', '###')]
    result = {}
    for sampling in samplings:
        model = V5Spotter(detector, reading, detector_hw=config['detector_hw'],
                          crop_kwargs={**config['crop'], 'sampling': sampling},
                          exposure=config.get('exposure', 'none')).eval()
        predictions = model.recognize_polygons(image, [p['poly'] for p in detections])
        for prediction, detection in zip(predictions, detections):
            prediction['score'] = detection['score']
        words = model.recognize_polygons(image, [gt['poly'] for _, gt in eligible])
        result[sampling] = dict(predictions=predictions, gt_crops=[
            dict(gt_index=i, text=gt['text'], prediction=word['text'], crop_error=word['crop_error'],
                 crop_strategy=word['crop_strategy'], crop_fallback_reason=word['crop_fallback_reason'],
                 crop_exposure_gain=word['crop_exposure_gain'])
            for (i, gt), word in zip(eligible, words)])
    return result


def run(detector_checkpoint, reading_checkpoint, output, samplings=('local_normal', 'polygon_boundary'),
        crop_pad=1., enhancement='full', exposure='none'):
    if not np.isfinite(crop_pad) or crop_pad <= 0 or enhancement not in ('full', 'no_denoise', 'identity'):
        raise ValueError('Positive crop padding and full/no_denoise/identity enhancement are required')
    config = plain(load_config('configs/v5/fusion_placement_dev.yaml'))
    rows = verified_rows(config)
    config.update(experiment=dict(id='connector_actual_development', role='fixed-weight crop sampling comparison'),
                  weights=dict(detector=str(detector_checkpoint), reading=str(reading_checkpoint)),
                  output=str(output), cases=list(samplings), enhancement=enhancement, exposure=exposure)
    config['crop']['pad'] = crop_pad
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='initializing', scope='intermediate detector development diagnosis; not final baseline/fusion acceptance',
                  config=config, weights={str(p): sha256(p) for p in (detector_checkpoint, reading_checkpoint)},
                  metric_scope='internal dev score-greedy polygon IoU>=0.5, lower-alnum exact text E2E; raw and lower-alnum GT-crop CER; not official test',
                  data_hashes={str(p): sha256(p) for p in [config['protocol'], *config['manifests'].values()]},
                  source_hashes={**reading_sources(), **detection_sources(),
                                 **{name: sha256(name) for name in ('tools/v5_eval_connector.py', 'tools/v5_eval_spotting.py',
                                                                    'seeread/v5/spotting.py', 'seeread/models/rectify.py')}},
                  results={})
    write_json(output / 'run.json', report)
    started = time.monotonic()
    try:
        device = require_cuda('cuda:0')
        torch.cuda.set_per_process_memory_fraction(.35, 0)
        detector, recognizer, enhancer, provenance = load_models(
            dict(detector=detector_checkpoint, reading=reading_checkpoint), device, allow_pretraining=True)
        if enhancement == 'no_denoise':
            if enhancer.denoise is None:
                raise ValueError('Denoise removal requires a stage2 enhancer')
            enhancer.denoise = None
        reading = ReadingPipeline(nn.Identity() if enhancement == 'identity' else enhancer, recognizer).requires_grad_(False).eval()
        report.update(status='running', provenance=provenance,
                      effective_enhancement=dict(module=type(reading.enhancer).__name__,
                                                 denoise_present=getattr(reading.enhancer, 'denoise', None) is not None))
        write_json(output / 'run.json', report)
        for illumination, samples in rows.items():
            evaluators = {key: SpottingEvaluator(iou_thr=.5) for key in samplings}
            words = {key: [] for key in evaluators}
            with (output / f'{illumination}.records.jsonl').open('x') as stream:
                for index, row in enumerate(samples):
                    image = rgb_tensor(Path(config['data_root']) / row['image'], device)
                    cases = compare_image(image, row['instances'], detector, reading, config, samplings)
                    for key, value in cases.items():
                        evaluators[key].add(value['predictions'], row['instances'])
                        words[key].extend(value['gt_crops'])
                        predictions = [dict(item, poly=np.asarray(item['poly']).tolist()) for item in value['predictions']]
                        stream.write(json.dumps(dict(image=row['image'], source_id=row['source_id'], sampling=key,
                                                     predictions=predictions, gt_crops=value['gt_crops'],
                                                     associations=association_diagnostics(predictions, row['instances'])),
                                                ensure_ascii=False, allow_nan=False)+'\n')
                    stream.flush()
                    if (index+1) % 10 == 0:
                        print(json.dumps(dict(illumination=illumination, images=index+1, expected=len(samples),
                                              elapsed_seconds=time.monotonic()-started)), flush=True)
                report['results'][illumination] = dict(complete=True, images=len(samples), cases={
                    key: dict(gt_crop_raw=metrics(words[key], 'raw'),
                              gt_crop_lower_alnum=metrics(words[key], 'lower_alnum'),
                              spotting=evaluators[key].summary()['all'],
                              gt_crop_failures=sum(bool(w['crop_error']) for w in words[key])) for key in words})
            write_json(output / 'run.json', report)
        report.update(status='complete', elapsed_seconds=time.monotonic()-started)
        write_json(output / 'run.json', report)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output / 'run.json', report)
        raise
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detector', type=Path, required=True)
    parser.add_argument('--reading', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sampling', nargs='+', choices=('local_normal', 'polygon_boundary', 'boundary_when_supported',
                                                        'boundary_with_hull_fallback'),
                        default=['local_normal', 'polygon_boundary'])
    parser.add_argument('--crop-pad', type=float, default=1.)
    parser.add_argument('--enhancement', choices=('full', 'no_denoise', 'identity'), default='full')
    parser.add_argument('--exposure', choices=('none', 'p95'), default='none')
    args = parser.parse_args()
    run(args.detector, args.reading, args.output, args.sampling, args.crop_pad, args.enhancement, args.exposure)
