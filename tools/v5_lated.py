"""LATeD MVP: drafts, whole-image inference, reviewed evaluation and paired reports."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DETECTOR = 'results/v5/detection/LATeD_geometry_fast_20261025/best.pt'
DEFAULT_READING = 'results/v5/fusion/read_connector_lora_20261025/best.pt'
CROP = dict(output_hw=(384, 384), center_points=32, pad=1.1, sampling='boundary_with_hull_fallback')


def bootstrap(runtime):
    sys.path.insert(0, str(ROOT))
    if runtime:
        runtime = Path(runtime).resolve()
        sys.path.insert(0, str(runtime)); sys.path.insert(0, str(runtime / 'tools'))
        import seeread.v5
        seeread.v5.__path__.append(str(ROOT / 'seeread/v5'))


def models(args):
    import torch
    from seeread.v5.recognition import ReadingPipeline
    from seeread.v5.spotting import V5Spotter
    from tools.v5_eval_spotting import load_models
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Run in an authorized single-GPU container')
    torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(.35)
    detector, rec, enhancer, provenance = load_models(dict(detector=args.detector, reading=args.reading), 'cuda:0')
    if not 2 <= args.max_len <= rec.base.decoder.config.max_position_embeddings:
        raise ValueError('Requested generation length exceeds decoder positional capacity')
    rec.max_len = args.max_len
    reading = ReadingPipeline(enhancer, rec).requires_grad_(False).eval()
    saved = torch.load(args.detector, map_location='cpu', weights_only=False)
    decoder = saved['config']['decoder']
    spotter = V5Spotter(detector, reading, rec_chunk=4, crop_kwargs=CROP, decode_kwargs=decoder).eval()
    return spotter, decoder, provenance


def rows_for(args):
    from seeread.v5.lated_annotations import labelled_rows, read_jsonl
    if args.sidecar:
        return labelled_rows(args.package, args.sidecar, 'dev', pilot=args.pilot)
    if args.image:
        return [dict(image=str(args.image), instances=[])]
    if 'test' in args.manifest.stem.lower():
        raise ValueError('MVP uses development only; final test requires a separately locked release')
    return read_jsonl(args.manifest)


def infer(args):
    import cv2
    import numpy as np
    from seeread.v5.lated_annotations import write_jsonl
    from seeread.v5.reading import sha256
    import seeread.v5.spotting as spotting_source
    import seeread.models.rectify as rectify_source
    from tools.v5_eval_spotting import rgb_tensor
    rows = rows_for(args)
    if args.limit:
        rows = rows[:args.limit]
    spotter, decoder, provenance = models(args)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'visualizations').mkdir()
    report = dict(status='running', scope='LATeD MVP development; not official test',
                  detector_sha256=sha256(args.detector), reading_sha256=sha256(args.reading),
                  max_len=args.max_len, detector_hw=[640, 640], crop=CROP, decoder=decoder,
                  package_sha256=sha256(args.package / 'protocol.json') if args.sidecar else None,
                  sidecar_sha256=sha256(args.sidecar) if args.sidecar else None, pilot=args.pilot,
                  source_sha256={str(p): sha256(p) for p in (Path(__file__), Path(spotting_source.__file__), Path(rectify_source.__file__))},
                  image_sha256={r['image']: sha256(args.root / r['image']) for r in rows}, provenance=provenance)
    (args.output / 'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    records = []
    with (args.output / 'records.jsonl').open('x') as stream:
        for row in rows:
            path = args.root / row['image']
            image = rgb_tensor(path, 'cuda:0')
            predictions = spotter.predict(image)
            eligible = [g for g in row['instances'] if g.get('ocr_eligible') and not g.get('ignore')]
            crops = spotter.recognize_polygons(image, [g['poly'] for g in eligible])
            record = dict(image=row['image'], predictions=[dict(p, poly=np.asarray(p['poly']).tolist()) for p in predictions],
                          gt_crops=[dict(p, id=g['id'], poly=np.asarray(p['poly']).tolist()) for p, g in zip(crops, eligible)])
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n'); stream.flush(); records.append(record)
            canvas = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            for p in predictions:
                poly = np.rint(p['poly']).astype(np.int32)
                cv2.polylines(canvas, [poly], True, (0, 255, 255), 2)
                label = f"{p['instance_id']}: {p['text']} ({p['score']:.2f})"
                cv2.putText(canvas, label, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
            name = hashlib.sha256(row['image'].encode()).hexdigest()[:20]
            cv2.imwrite(str(args.output / 'visualizations' / f'{name}.jpg'), canvas)
            print(json.dumps(dict(images=len(records), expected=len(rows))), flush=True)
    if args.sidecar and not args.limit:
        from seeread.v5.lated_evaluation import evaluate_records
        report['evaluation'] = evaluate_records(rows, records)
    cards = []
    for record in records:
        name = hashlib.sha256(record['image'].encode()).hexdigest()[:20]
        labels = '<br>'.join(html.escape(f"{p['instance_id']}: {p['text']} | score={p['score']:.3f} | crop={p['crop_error']}")
                             for p in record['predictions'])
        cards.append(f'<article><h3>{html.escape(record["image"])}</h3><img src="visualizations/{name}.jpg">'
                     f'<p>{labels or "No detections"}</p></article>')
    (args.output / 'preview.html').write_text('<!doctype html><meta charset="utf-8"><title>LATeD MVP</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:auto}img{max-width:100%}article{padding:20px;border-bottom:1px solid #aaa}</style>'
        '<h1>LATeD MVP：整图多实例识别</h1><p>固定基线输出；未审核转写前不报告真实识别成绩。</p>' + ''.join(cards))
    report.update(status='complete', images=len(records))
    (args.output / 'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


def drafts(args):
    import numpy as np
    from seeread.v5.lated_annotations import verify_package, write_jsonl
    from seeread.v5.lated_annotations import render_pages
    from tools.v5_eval_spotting import rgb_tensor
    from seeread.v5.reading import sha256
    protocol, tasks = verify_package(args.package)
    spotter, _, _ = models(args)
    grouped = {}
    for task in tasks:
        grouped.setdefault(task['image'], []).append(task)
    output = args.package / 'drafts.jsonl'
    with output.open('x') as stream:
        for index, (path, items) in enumerate(grouped.items()):
            image = rgb_tensor(Path(protocol['data_root']) / path, 'cuda:0')
            predictions = spotter.recognize_polygons(image, [t['poly'] for t in items])
            for task, p in zip(items, predictions):
                stream.write(json.dumps(dict(id=task['id'], text=p['text'], crop_error=p['crop_error'],
                                             status='machine_draft_not_ground_truth'), ensure_ascii=False)+'\n')
            stream.flush()
            if (index+1) % 20 == 0:
                print(json.dumps(dict(images=index+1, total=len(grouped))), flush=True)
    (args.package / 'drafts_provenance.json').write_text(json.dumps(dict(reading_sha256=sha256(args.reading),
        max_len=args.max_len, crop=CROP, package_sha256=sha256(args.package/'protocol.json'), status='complete'), indent=2)+'\n')
    render_pages(args.package)
    return dict(status='complete', images=len(grouped), drafts=len(tasks))


def prepare_crops(args):
    import cv2
    import numpy as np
    from seeread.v5.lated_annotations import labelled_rows, read_jsonl, verify_package, write_jsonl
    from seeread.v5.spotting import polygon_crop
    from seeread.v5.reading import WordCropDataset, check_disjoint, sha256
    from tools.v5_eval_spotting import rgb_tensor
    protocol, _ = verify_package(args.package)
    # Review gate is checked before creating any crop or training artifact.
    sources = {s: labelled_rows(args.package, args.sidecar, s) for s in ('train', 'dev')}
    predictions = None
    if args.predictions:
        from seeread.v5.lated_evaluation import reliable_training_matches
        saved = json.loads((args.predictions/'run.json').read_text())
        records = read_jsonl(args.predictions/'records.jsonl')
        if saved['status'] != 'complete' or [r['image'] for r in records] != [r['image'] for r in sources['train']]:
            raise ValueError('Prediction adaptation requires the complete selected training images only')
        if saved['image_sha256'] != {r['image']: sha256(args.root/r['image']) for r in sources['train']}:
            raise ValueError('Training prediction image bytes differ')
        predictions = {r['image']: r['predictions'] for r in records}
    args.output.mkdir(parents=True, exist_ok=False)
    image_dir = args.root / 'derived/v5' / args.output.name
    image_dir.mkdir(parents=True, exist_ok=False)
    failures, counts = [], {}
    for split, rows in sources.items():
        write_jsonl(args.output / f'{split}.scenes.jsonl', rows)
        words = []
        for row in rows:
            image = rgb_tensor(args.root / row['image'], 'cpu')
            instances = row['instances']
            if predictions is not None and split == 'train':
                boxes = predictions[row['image']]
                instances = [dict(row['instances'][gi], poly=boxes[pi]['poly'])
                             for pi, gi in reliable_training_matches(boxes, row['instances'])]
            for gt in instances:
                if not gt['ocr_eligible']:
                    continue
                crop = polygon_crop(image, gt['poly'], **CROP)
                if crop.crop is None:
                    failures.append(dict(id=gt['id'], split=split, crop_error=crop.reason)); continue
                identity = hashlib.sha256(gt['id'].encode()).hexdigest()
                path = image_dir / f'{identity}.png'
                rgb = crop.crop[0].clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().numpy()
                if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                    raise OSError(path)
                words.append(dict(id=gt['id'], image=str(path.relative_to(args.root)), text=gt['text'], split=split,
                                  source='LATeD_human_reviewed', dataset='LATeD_MVP', source_group=row['image'],
                                  sha256={'image': sha256(path)}))
        write_jsonl(args.output / f'{split}.jsonl', words); counts[split] = len(words)
    datasets = [WordCropDataset(args.output/f'{s}.jsonl', args.root, s) for s in ('train', 'dev')]
    check_disjoint(*datasets)
    result = dict(status='complete', package_sha256=sha256(args.package/'protocol.json'), sidecar_sha256=sha256(args.sidecar),
                  crop=CROP, counts=counts, failures=failures, manifests={s:sha256(args.output/f'{s}.jsonl') for s in sources},
                  training_crops='reliable_train_predictions' if predictions is not None else 'GT',
                  prediction_report_sha256=sha256(args.predictions/'run.json') if args.predictions else None,
                  selection='valid PNG-crop dev CER; full-scene evaluation keeps every failed GT crop')
    (args.output/'protocol.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


def protocol_score(args):
    from seeread.v5.lated_annotations import read_jsonl
    from seeread.v5.lated_evaluation import image_counts, summary
    from seeread.v5.reading import sha256
    source, records = read_jsonl(args.manifest), read_jsonl(args.records)
    if 'test' in args.manifest.stem.lower():
        raise ValueError('Official500 test remains locked; this command validates the development protocol')
    if [r['image'] for r in source] != [r['image'] for r in records]:
        raise ValueError('Prediction/source image order or completeness differs')
    total = Counter()
    for row, record in zip(source, records):
        if 'ground_truth' in record:
            if [g['poly'] for g in row['instances']] != [g['poly'] for g in record['ground_truth']]:
                raise ValueError('Prediction records are not in the original source coordinate frame')
        counts, _ = image_counts(record['predictions'], row['instances']); total.update(counts)
    result = dict(status='complete', protocol='ctw_voc_matching_port_for_lated',
                  author_implementation=False, source_commit='2126a660d3baf3e84a76e77352597576519c35a2',
                  manifest_sha256=sha256(args.manifest), records_sha256=sha256(args.records), **summary(total))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f: json.dump(result, f, indent=2)
    return result


def compare(args):
    from seeread.v5.lated_annotations import read_jsonl
    from seeread.v5.lated_evaluation import paired_comparison
    a, b = [json.loads((p/'run.json').read_text()) for p in (args.reference, args.candidate)]
    if not all(r.get('evaluation') for r in (a,b)):
        raise ValueError('Comparison requires two complete human-reviewed development evaluations')
    for k in ('detector_sha256','decoder','crop','max_len','detector_hw','sidecar_sha256','package_sha256','image_sha256','pilot','source_sha256'):
        if a[k] != b[k]: raise ValueError(f'Controlled Read comparison changed {k}')
    reference_records, candidate_records = [read_jsonl(p/'records.jsonl') for p in (args.reference,args.candidate)]
    if len(reference_records) != a['images'] or len(candidate_records) != b['images']:
        raise ValueError('Prediction record count differs from completed reports')
    for left,right in zip(reference_records,candidate_records):
        boxes = lambda row:[(p['poly'],p['score']) for p in row['predictions']]
        if left['image'] != right['image'] or boxes(left) != boxes(right):
            raise ValueError('Fixed-detector comparison changed boxes/scores')
    if a['status'] != 'complete' or b['status'] != 'complete' or a['images'] != b['images']:
        raise ValueError('Both complete evaluations are required')
    result = paired_comparison(a['evaluation'], b['evaluation'])
    with args.output.open('x') as f: json.dump(result,f,indent=2)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',type=Path,default=Path('results/v5/runtime/lated_geometry_fast_20261025'))
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('infer','drafts'):
        s=sub.add_parser(name);s.add_argument('--detector',type=Path,default=Path(DEFAULT_DETECTOR));s.add_argument('--reading',type=Path,default=Path(DEFAULT_READING))
        s.add_argument('--max-len',type=int,default=128);s.add_argument('--package',type=Path,default=Path('data/derived/v5/lated_mvp_20261026'))
        if name=='infer':
            s.add_argument('--root',type=Path,default=Path('data'));s.add_argument('--manifest',type=Path,default=Path('data/annotations/v5/detection/latedspot_val.jsonl'))
            s.add_argument('--pilot',action='store_true');s.add_argument('--image',type=Path);s.add_argument('--sidecar',type=Path);s.add_argument('--output',type=Path,required=True);s.add_argument('--limit',type=int,default=0)
    s=sub.add_parser('prepare-crops');s.add_argument('--package',type=Path,required=True);s.add_argument('--sidecar',type=Path,required=True);s.add_argument('--root',type=Path,default=Path('data'));s.add_argument('--output',type=Path,required=True);s.add_argument('--predictions',type=Path)
    s=sub.add_parser('protocol');s.add_argument('--manifest',type=Path,required=True);s.add_argument('--records',type=Path,required=True);s.add_argument('--output',type=Path,required=True)
    s=sub.add_parser('compare');s.add_argument('--reference',type=Path,required=True);s.add_argument('--candidate',type=Path,required=True);s.add_argument('--output',type=Path,required=True)
    args=p.parse_args();bootstrap(args.runtime if args.command in ('infer','drafts') else None)
    result={'infer':infer,'drafts':drafts,'prepare-crops':prepare_crops,'protocol':protocol_score,'compare':compare}[args.command](args)
    print(json.dumps({k:v for k,v in result.items() if k in ('status','images','drafts','detection','latin_e2e','gain_gate','e2e_F1_delta','gt_CER_delta')}, ensure_ascii=False),flush=True)


if __name__=='__main__':main()
