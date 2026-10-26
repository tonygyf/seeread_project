"""Add human-view Self-DACE previews and raw/enhanced OCR without changing GT."""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.v5_lated import bootstrap, models, DEFAULT_DETECTOR, DEFAULT_READING, CROP


def run(args):
    bootstrap(args.runtime)
    import cv2
    import torch
    from torch import nn
    from torch.nn import functional as F
    from seeread.v5.lated_annotations import verify_package, read_jsonl, render_pages
    from seeread.v5.reading import sha256
    from seeread.v5.recognition import ReadingPipeline
    from seeread.v5.spotting import V5Spotter
    from tools.v5_eval_spotting import rgb_tensor

    protocol, tasks = verify_package(args.package)
    binding = sha256(args.package / 'protocol.json')
    parent = json.loads((args.package / 'drafts_provenance.json').read_text())
    if parent['reading_sha256'] != sha256(args.reading) or parent['max_len'] != args.max_len or parent['crop'] != json.loads(json.dumps(CROP)):
        raise ValueError('Existing enhanced OCR drafts differ from the selected weights/crop/generation protocol')
    existing = {row['id']: row for row in read_jsonl(args.package / 'drafts.jsonl')}
    if set(existing) != {t['id'] for t in tasks}:
        raise ValueError('Enhanced drafts do not cover the frozen package exactly')
    spotter, _, _ = models(args)
    raw = V5Spotter(spotter.detector, ReadingPipeline(nn.Identity(), spotter.reading.recognizer),
                    rec_chunk=4, crop_kwargs=CROP).eval()
    groups = OrderedDict()
    for task in sorted(tasks, key=lambda t: not t['pilot']):
        groups.setdefault(task['image'], []).append(task)
    output = args.package / 'review_assists.jsonl'
    if output.exists():
        raise FileExistsError('Review assists already exist; preserve their provenance')
    (args.package / 'crops_lle').mkdir(exist_ok=False)
    report = dict(status='running', package_sha256=binding, reading_sha256=parent['reading_sha256'],
                  raw_ocr='same recognizer; enhancer bypassed; same polygon rectification',
                  enhanced_ocr='existing full Self-DACE+recognizer drafts; unchanged',
                  lle='same frozen Self-DACE, aspect-preserving preview, longest side<=768; human aid only',
                  max_len=args.max_len, crop=CROP, images=0, instances=0)
    report_path = args.package / 'review_assists_provenance.json'
    def publish():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        render_pages(args.package)
    publish()
    try:
        with output.open('x') as stream:
            for index, (path, items) in enumerate(groups.items()):
                image = rgb_tensor(Path(protocol['data_root']) / path, 'cuda:0')
                predictions = raw.recognize_polygons(image, [task['poly'] for task in items])
                for task, prediction in zip(items, predictions):
                    crop_path = args.package / task['crop']
                    crop = rgb_tensor(crop_path, 'cuda:0')[None]
                    h, w = crop.shape[-2:]
                    factor = min(1., 768 / max(h, w))
                    if factor < 1:
                        crop = F.interpolate(crop, size=(max(1,round(h*factor)),max(1,round(w*factor))), mode='bilinear',align_corners=False)
                    with torch.inference_mode():
                        enhanced = spotter.reading.enhancer(crop)
                    pixels = enhanced[0].clamp(0,1).permute(1,2,0).mul(255).round().byte().cpu().numpy()
                    target = Path('crops_lle') / crop_path.name
                    if not cv2.imwrite(str(args.package / target), cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR)):
                        raise OSError(target)
                    before = existing[task['id']]
                    record = dict(id=task['id'], raw_text=prediction['text'], enhanced_text=before['text'],
                                  raw_crop_error=prediction['crop_error'], enhanced_crop_error=before['crop_error'],
                                  lle_crop=str(target), status='machine_assist_not_ground_truth')
                    stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                    report['instances'] += 1
                stream.flush(); report['images'] = index + 1
                if index == 0 or index == 19 or (index + 1) % 25 == 0:
                    publish();print(json.dumps(dict(images=index+1,instances=report['instances'],pilot_ready=index>=19)),flush=True)
        report['status'] = 'complete'
        if sha256(args.package/'protocol.json') != binding:
            raise ValueError('Frozen package changed while generating review assists')
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        publish()
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',type=Path,default=Path('results/v5/runtime/lated_geometry_fast_20261025'))
    p.add_argument('--package',type=Path,default=Path('data/derived/v5/lated_mvp_20261026'))
    p.add_argument('--reading',type=Path,default=Path(DEFAULT_READING));p.add_argument('--detector',type=Path,default=Path(DEFAULT_DETECTOR))
    p.add_argument('--max-len',type=int,default=128)
    result=run(p.parse_args());print(json.dumps({k:result[k] for k in ('status','images','instances')}),flush=True)
