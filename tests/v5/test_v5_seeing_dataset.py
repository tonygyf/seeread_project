"""Indexed detection data tests; temporary tiny fixtures, no CUDA/network."""
from pathlib import Path
import hashlib
import json
import pickle
import sys
import tempfile
from unittest.mock import patch

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from seeread.v5.seeing_dataset import SeeingDataset, collate_seeing, _sha256


def fixture(root, count=3, filename='source_train.jsonl'):
    rows = []
    for index in range(count):
        image = np.zeros((40, 80, 3), np.uint8)
        image[:, :] = [20 + index, 70 + index, 200 + index]  # BGR
        cv2.rectangle(image, (8, 12), (60, 28), (200, 220, 250), -1)
        path = root / f'image{index}.png'
        cv2.imwrite(str(path), image)
        rows.append(dict(image=path.name, instances=[
            dict(poly=[[8, 12], [60, 12], [60, 28], [8, 28]], text='词' + str(index), ignore=False),
            dict(poly=[[64, 0], [79, 0], [79, 39], [64, 39]], text='###', ignore=True)],
            meta=dict(illum_kind='lit', id=str(index))))
    manifest = root / filename
    with open(manifest, 'w') as stream:
        for index, row in enumerate(rows):
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            if index == 0:
                stream.write('\n')
    return manifest, rows


def dataset(root, manifest, **kwargs):
    return SeeingDataset(manifest, root=root, image_hw=(80, 120), feature_hw=(20, 30),
                         index_cache_dir=root / 'cache', **kwargs)


def test_offset_mmap_random_access_and_same_size_cache_invalidation():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest, rows = fixture(root)
        # Building/using the JSONL loader must not depend on Path.read_text.
        with patch.object(Path, 'read_text', side_effect=AssertionError('whole manifest text loaded')):
            first = dataset(root, manifest)
            assert isinstance(first._offsets, np.memmap) and len(first) == 3
            assert [first[i]['metadata']['id'] for i in (2, 0, 1)] == ['2', '0', '1']
            cached = dataset(root, manifest)
            assert cached.provenance['index_cache_hit']
        stat = manifest.stat()
        old_hash = first.provenance['manifest_sha256']
        raw = manifest.read_bytes().replace(b'"id": "1"', b'"id": "9"')
        manifest.write_bytes(raw)
        import os
        os.utime(manifest, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        changed = dataset(root, manifest)
        assert not changed.provenance['index_cache_hit']
        assert changed.provenance['manifest_sha256'] != old_hash
        assert changed[1]['metadata']['id'] == '9'
        # Corrupt index bytes also invalidate the cache, independently of JSONL.
        with open(changed._index_path, 'ab') as stream:
            stream.write(b'corruption')
        repaired = dataset(root, manifest)
        assert not repaired.provenance['index_cache_hit'] and repaired[2]['index'] == 2


def test_resize_rgb_polygon_ignore_and_collation_contract():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest, rows = fixture(root)
        ds = dataset(root, manifest)
        sample = ds[0]
        expected = np.asarray(rows[0]['instances'][0]['poly']) * [1.5, 2]
        assert sample['original_instances'] == rows[0]['instances']
        assert np.allclose(sample['transformed_instances'][0]['poly'], expected)
        assert sample['original_hw'] == (40, 80)
        assert sample['image'].shape == (3, 80, 120)
        assert torch.allclose(sample['image'][:, 1, 1], torch.tensor([200, 70, 20]) / 255.)
        assert sample['targets']['text'].shape == (20, 30)
        assert sample['targets']['text'].dtype == torch.int64
        assert sample['targets']['valid'].dtype == torch.bool
        assert not sample['targets']['valid'][:, 26:].any()
        assert sample['targets']['center'].sum() > 0
        batch = collate_seeing([sample, ds[1]])
        assert batch['image'].shape == (2, 3, 80, 120)
        assert batch['targets']['height'].shape == (2, 20, 30)
        assert batch['image_path'] == ['image0.png', 'image1.png']
        # meta.illum_kind=lit MUST NOT trigger legacy online darkening.
        assert sample['illumination'] == {'policy': 'original'} and not sample['augmentation']


def test_epoch_sampler_augmentation_and_fixed_illumination_are_independent():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest, _ = fixture(root)
        ds = dataset(root, manifest, illumination='fixed_synth',
                     augment={'flip_p': 1, 'noise_p': 1, 'noise_std': .02}, seed=7)
        a, repeated, later = ds[(0, 0)], ds[(0, 0)], ds[(1, 0)]
        assert torch.equal(a['image'], repeated['image'])
        assert not torch.equal(a['image'], later['image'])
        assert a['illumination'] == later['illumination']
        expected = np.array([[12., 24.], [90., 24.], [90., 56.], [12., 56.]])
        expected[:, 0] = 119 - expected[:, 0]
        assert np.allclose(a['transformed_instances'][0]['poly'], expected)
        plain = dataset(root, manifest, illumination='fixed_synth', seed=7)
        plain.set_epoch(2)
        assert torch.equal(plain[0]['image'], plain[(9, 0)]['image'])
        restored = pickle.loads(pickle.dumps(plain))
        assert torch.equal(restored[0]['image'], plain[0]['image'])


def test_scale_crop_keeps_pixels_polygons_and_ignore_masks_aligned():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest, rows = fixture(root, count=1)
        # A white rectangle independently exposes image/target misalignment.
        image = np.zeros((40, 80, 3), np.uint8)
        cv2.rectangle(image, (8, 12), (60, 28), (255, 255, 255), -1)
        cv2.imwrite(str(root / 'image0.png'), image)
        for factor in (.75, 1.5):
            ds = SeeingDataset(manifest, root=root, image_hw=(40, 80), feature_hw=(40, 80),
                               index_cache_dir=root / 'cache', seed=7,
                               augment=dict(scale_crop_p=1., scale_min=factor, scale_max=factor))
            a, repeated, later = ds[(0, 0)], ds[(0, 0)], ds[(1, 0)]
            assert torch.equal(a['image'], repeated['image'])
            assert not torch.equal(a['image'], later['image'])
            assert a['original_instances'] == rows[0]['instances']
            foreground = a['image'][0].numpy() > .5
            target = a['targets']['text'].numpy().astype(bool)
            assert (foreground & target).sum() / (foreground | target).sum() > .9
            ignored = np.zeros((40, 80), np.uint8)
            cv2.fillPoly(ignored, [np.rint(a['transformed_instances'][1]['poly']).astype(np.int32)], 1)
            assert np.array_equal(a['targets']['valid'].numpy(), ignored == 0)
            assert torch.isfinite(a['targets']['height']).all()
            assert a['targets']['center'].sum() > 0


def test_scale_crop_is_train_only_and_requires_a_positive_ordered_range():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest, _ = fixture(root, filename='source_dev.jsonl')
        import pytest
        with pytest.raises(ValueError, match='Random training augmentation'):
            dataset(root, manifest, split='dev', augment=dict(scale_crop_p=1.))
        for low, high in ((0., 1.), (2., 1.)):
            with pytest.raises(ValueError, match='scale_min'):
                dataset(root, manifest, split='dev', augment=dict(scale_min=low, scale_max=high))


def test_fingerprints_and_rooted_paths_are_enforced():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest, rows = fixture(root)
        fingerprint = root / 'image_fingerprints.jsonl'
        fingerprint.write_text(json.dumps(dict(path='image0.png', sha256=_sha256(root / 'image0.png'), width=80, height=40)) + '\n')
        ds = dataset(root, manifest, require_fingerprint=True)
        assert ds[0]['image_fingerprint_verified']
        try:
            ds[1]
        except ValueError as error:
            assert 'No image fingerprint' in str(error)
        else:
            raise AssertionError('Required but unavailable fingerprint was ignored')
        cv2.imwrite(str(root / 'image0.png'), np.zeros((40, 80, 3), np.uint8))
        try:
            ds[0]
        except ValueError as error:
            assert 'fingerprint mismatch' in str(error)
        else:
            raise AssertionError('Modified image accepted')
        for path in ('../outside.png', '/outside.png'):
            try:
                ds._path(path)
            except ValueError:
                pass
            else:
                raise AssertionError('Non-rooted image accepted')
        with tempfile.TemporaryDirectory() as outside:
            (root / 'escape').symlink_to(outside, target_is_directory=True)
            try:
                ds._path('escape/image.png')
            except ValueError:
                pass
            else:
                raise AssertionError('Escaping symlink accepted')


def test_test_data_is_rejected_by_role_filename_and_registered_content():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest, rows = fixture(root, filename='source_test.jsonl')
        try:
            dataset(root, manifest)
        except ValueError:
            pass
        else:
            raise AssertionError('Test filename accepted for train')
        renamed = root / 'renamed_train.jsonl'
        renamed.write_bytes(manifest.read_bytes())
        protocol = root / 'provenance.json'
        protocol.write_text(json.dumps({'frozen_manifests': {'original_test': {
            'path': str(manifest), 'sha256': _sha256(manifest), 'role': 'final_test_registration_only'}}}))
        try:
            dataset(root, renamed)
        except ValueError:
            pass
        else:
            raise AssertionError('Renamed registered test contents accepted for train')
        protocol.unlink()
        rows[0]['split'] = 'test'
        renamed.write_text(json.dumps(rows[0]) + '\n')
        ds = dataset(root, renamed)
        try:
            ds[0]
        except ValueError:
            pass
        else:
            raise AssertionError('Row explicitly marked test accepted')


def test_full_reference_is_sha_bound_relocated_and_mlt_is_not_substituted():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ann = root / 'annotations'
        ann.mkdir()
        manifest, _ = fixture(root)
        source = ann / 'synthtext_train.jsonl'
        source.write_bytes(manifest.read_bytes())
        reference = root / 'synthtext_train.reference.json'
        descriptor = dict(source_path='/unavailable/host/data/annotations/synthtext_train.jsonl',
                          source_sha256=_sha256(source), source_bytes=source.stat().st_size)
        reference.write_text(json.dumps(descriptor))
        ds = dataset(root, reference)
        assert len(ds) == 3 and ds.manifest == source
        assert ds[2]['metadata']['id'] == '2'
        descriptor['source_sha256'] = '0' * 64
        reference.write_text(json.dumps(descriptor))
        try:
            dataset(root, reference)
        except ValueError as error:
            assert 'SHA256' in str(error)
        else:
            raise AssertionError('Wrong full reference SHA accepted')
        mlt = root / 'mlt17.prerequisite.json'
        mlt.write_text(json.dumps({'status': 'missing_prerequisite', 'usable_for_paper_recipe': False}))
        try:
            dataset(root, mlt)
        except FileNotFoundError as error:
            assert 'MLT' in str(error)
        else:
            raise AssertionError('Unavailable MLT silently substituted')


if __name__ == '__main__':
    torch.set_num_threads(2)
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
            print('PASS', name)
