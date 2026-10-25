"""Prepare independent train crops through the deployed connector; keep dev scenes held out."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from seeread.data.lowlight_synth import darken
from seeread.v5.reading import WordCropDataset, check_disjoint, sha256
from seeread.v5.spotting import polygon_crop


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def select_train(candidate, train, dev, test):
    by_image = {row['image']: row for row in candidate}
    if len(by_image) != len(candidate):
        raise ValueError('Duplicate candidate scene')
    selected = [by_image[row['image']] for row in train]
    selected_ids = {row['image'] for row in selected}
    if len(selected_ids) != len(selected) or selected_ids & {r['image'] for r in dev + test}:
        raise ValueError('Training scenes overlap held-out scenes or are duplicated')
    return selected


def prepare(root, out):
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    paths = dict(candidate=root / 'annotations/v5/candidates/totaltext_20261023/totaltext_train.jsonl',
                 train=root / 'annotations/v5/detection/totaltext_train.jsonl',
                 dev=root / 'annotations/v5/fusion/totaltext_dev_bright.jsonl',
                 dark=root / 'annotations/v5/fusion/totaltext_dev_dark.jsonl',
                 test=root / 'annotations/v5/detection/totaltext_test.jsonl')
    source = {key: rows(path) for key, path in paths.items()}
    train = select_train(source['candidate'], source['train'], source['dev'], source['test'])
    train_hashes = {sha256(root / row['image']) for row in train}
    heldout_hashes = {sha256(root / row['image']) for row in source['dev'] + source['test']}
    if train_hashes & heldout_hashes:
        raise ValueError('Training image bytes overlap held-out scenes')
    out.mkdir(parents=True, exist_ok=False)
    image_dir = root / 'derived/v5/connector_reading_20261025'
    image_dir.mkdir(parents=True, exist_ok=False)
    crop_cfg = dict(output_hw=(384, 384), center_points=32, pad=1.1, sampling='boundary_when_supported')
    counts, failures = {}, []
    for split, scenes in (('train', train), ('dev', source['dark'])):
        count = 0
        with (out / f'{split}.jsonl').open('x') as stream:
            for index, row in enumerate(scenes):
                bgr = cv2.imread(str(root / row['image']), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                if bgr is None:
                    raise FileNotFoundError(row['image'])
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                variants = {'dark': rgb}
                if split == 'train':
                    low, _ = darken(rgb, np.random.default_rng((20261025, index, 1)), level='dark', mode='physical')
                    variants = {'bright': rgb, 'dark': low}
                for illumination, pixels in variants.items():
                    image = torch.from_numpy(np.ascontiguousarray(pixels)).permute(2, 0, 1).float() / 255
                    for instance, gt in enumerate(row['instances']):
                        if gt.get('ignore') or gt.get('text', '') in ('', '###'):
                            continue
                        crop = polygon_crop(image, gt['poly'], **crop_cfg)
                        identifier = f'{split}_{index:04d}_{instance:03d}_{illumination}'
                        if crop.crop is None:
                            failures.append(dict(id=identifier, image=row['image'], text=gt['text'], reason=crop.reason))
                            continue
                        pixels = crop.crop[0].clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().numpy()
                        target = image_dir / f'{identifier}.png'
                        if not cv2.imwrite(str(target), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)):
                            raise OSError(target)
                        record = dict(id=identifier, image=str(target.relative_to(root)), text=gt['text'],
                                      source='TotalText_connector', source_group=row.get('source_id', row['image']),
                                      split=split, dataset='connector_reading', scene_group_verified=True,
                                      illumination=illumination, sha256={'image': sha256(target)})
                        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                        count += 1
                if (index + 1) % 100 == 0:
                    print(json.dumps(dict(split=split, scenes=index + 1, words=count)), flush=True)
        counts[split] = count
    datasets = [WordCropDataset(out / f'{split}.jsonl', root, split) for split in ('train', 'dev')]
    check_disjoint(*datasets)
    protocol = dict(role='fusion crop-domain adaptation; independent baselines unchanged', crop=crop_cfg,
                    seed=20261025, train_scenes=len(train), dev_scenes=len(source['dark']), counts=counts,
                    training='paired bright/dark crops, equal weight; one fixed physical darkening per train scene',
                    selection='dark valid-crop raw CER; final connector evaluation includes all failed GT crops',
                    storage='rounded uint8 PNG; final spotting evaluates original float crops', failures=failures,
                    annotation_status='corrected train-only source_candidate; not official benchmark reproduction',
                    sources={k: dict(path=str(p), sha256=sha256(p)) for k, p in paths.items()},
                    image_byte_overlap=0, source_sha256=sha256(__file__),
                    manifests={split: sha256(out / f'{split}.jsonl') for split in counts})
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(counts=counts, failed_crops=len(failures), train_scenes=len(train))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, default=Path('data/annotations/v5/connector_reading_20261025'))
    args = parser.parse_args()
    prepare(args.root, args.out)
