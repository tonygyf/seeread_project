"""Prepare the independent 200-train/100-dev LATeD review package (never test)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from seeread.v5.lated_annotations import read_jsonl, render_pages, select_training, write_jsonl
from seeread.v5.reading import sha256


def prepare(root, train, dev, output, train_images=200, pilot_images=10):
    root, output = Path(root).resolve(), Path(output)
    train, dev = Path(train).resolve(), Path(dev).resolve()
    if 'test' in train.stem.lower() or 'test' in dev.stem.lower():
        raise ValueError('Test images cannot enter the annotation pilot')
    trains = select_training(read_jsonl(train), root, train_images, seed=0)
    devs = read_jsonl(dev)
    if len(devs) != 100:
        raise ValueError('Expected the frozen complete 100-image LATeD development set')
    sources = {split: dict(path=str(p), sha256=sha256(p)) for split, p in [('train', train), ('dev', dev)]}
    fingerprints = {r['image']: sha256(root / r['image']) for r in trains + devs}
    if {fingerprints[r['image']] for r in trains} & {fingerprints[r['image']] for r in devs}:
        raise ValueError('Selected training and development image bytes overlap')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'crops').mkdir(); (output / 'contexts').mkdir()
    tasks = []
    for split, rows in [('train', trains), ('dev', devs)]:
        for index, row in enumerate(rows):
            image = cv2.imread(str(root / row['image']), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if image is None:
                raise FileNotFoundError(row['image'])
            h, w = image.shape[:2]
            context = image.copy()
            context_id = hashlib.sha256(row['image'].encode()).hexdigest()[:20]
            for number, ins in enumerate(row['instances']):
                poly = np.asarray(ins['poly'], np.float32)
                cv2.polylines(context, [np.rint(poly).astype(np.int32)], True, (255, 255, 0), 2)
                cv2.putText(context, str(number), tuple(np.rint(poly[0]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                if ins.get('ignore'):
                    continue
                crop_id = hashlib.sha256(ins['id'].encode()).hexdigest()[:24]
                x0, y0 = np.maximum(np.floor(poly.min(0) - 8).astype(int), 0)
                x1, y1 = np.minimum(np.ceil(poly.max(0) + 8).astype(int), [w, h])
                if x1 <= x0 or y1 <= y0:
                    raise ValueError(f'Empty annotation preview: {ins["id"]}')
                if not cv2.imwrite(str(output / 'crops' / f'{crop_id}.png'), image[y0:y1, x0:x1]):
                    raise OSError('Could not save review crop')
                tasks.append(dict(id=ins['id'], image=row['image'], split=split, image_sha256=fingerprints[row['image']],
                                  source_sha256=sources[split]['sha256'], poly=ins['poly'], instance_index=number,
                                  crop=f'crops/{crop_id}.png', context=f'contexts/{context_id}.jpg',
                                  pilot=index < pilot_images, language='unknown', status='pending'))
            scale = min(1., 1280 / max(h, w))
            context = cv2.resize(context, (max(1, round(w * scale)), max(1, round(h * scale))))
            if not cv2.imwrite(str(output / 'contexts' / f'{context_id}.jpg'), context):
                raise OSError('Could not save context preview')
    ids = [t['id'] for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError('Source instance IDs are not unique')
    write_jsonl(output / 'tasks.jsonl', tasks)
    protocol = dict(schema='lated_transcription_v1', seed=0, data_root=str(root), sources=sources,
                    tasks_sha256=sha256(output / 'tasks.jsonl'), images={'train': len(trains), 'dev': len(devs)},
                    instances={split: sum(t['split'] == split for t in tasks) for split in sources},
                    pilot_images_per_split=pilot_images, selection='SHA256 content groups, seeded hash order; whole groups only',
                    test_policy='Official500 test excluded; original annotations/images untouched',
                    ocr_scope='Human-reviewed Latin lines only; detection retains every source instance')
    (output / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    render_pages(output)
    print(json.dumps(protocol, ensure_ascii=False), flush=True)
    return protocol


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('data'))
    p.add_argument('--train', type=Path, default=Path('data/annotations/v5/detection/latedspot_train.jsonl'))
    p.add_argument('--dev', type=Path, default=Path('data/annotations/v5/detection/latedspot_val.jsonl'))
    p.add_argument('--output', type=Path, default=Path('data/derived/v5/lated_mvp_20261026'))
    p.add_argument('--render-only', action='store_true')
    a = p.parse_args()
    render_pages(a.output) if a.render_only else prepare(a.root, a.train, a.dev, a.output)
