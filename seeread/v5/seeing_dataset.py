"""Indexed, manifest-bound detection input for independent V5 training.

JSONL rows are read by byte offset. The index is an mmap, bound to a fresh
streaming SHA256 of the manifest and its own checksum. Dataset construction
never decodes all JSON rows or materializes the SynthText manifest in memory.
Resize is an explicit rectangular stretch, followed by configured train-only
augmentation; development images are never randomly cropped or scaled.
"""
from __future__ import annotations
from array import array
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from seeread.data.lowlight_synth import darken
from seeread.v5.seeing_data import targets_from_polygons

INDEX_FORMAT = 'seeread_detection_offsets_v1'


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _json(path):
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)


def _index(manifest, cache_dir):
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(manifest).encode()).hexdigest()
    index_path, meta_path = cache / (key + '.npy'), cache / (key + '.json')
    with open(cache / (key + '.lock'), 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if index_path.is_file() and meta_path.is_file():
            try:
                meta = _json(meta_path)
                # Rehash even when stat metadata matches. A same-length edit or
                # restored mtime cannot quietly reuse another manifest's index.
                if (meta['format'] == INDEX_FORMAT and meta['manifest_sha256'] == _sha256(manifest)
                        and meta['index_sha256'] == _sha256(index_path)):
                    offsets = np.load(index_path, mmap_mode='r', allow_pickle=False)
                    if offsets.dtype == np.dtype('uint64') and offsets.shape == (meta['rows'],):
                        return offsets, meta, index_path, True
            except (KeyError, ValueError, OSError, json.JSONDecodeError):
                pass
        offsets = array('Q')
        digest, position, blanks = hashlib.sha256(), 0, 0
        with open(manifest, 'rb') as stream:
            for line in stream:
                digest.update(line)
                if line.strip():
                    offsets.append(position)
                else:
                    blanks += 1
                position += len(line)
        if not offsets:
            raise ValueError('Empty detection manifest')
        temporary = index_path.with_suffix('.npy.tmp')
        with open(temporary, 'wb') as stream:
            np.save(stream, np.frombuffer(offsets, dtype=np.uint64), allow_pickle=False)
        temporary.replace(index_path)
        meta = dict(format=INDEX_FORMAT, manifest=str(manifest), manifest_sha256=digest.hexdigest(),
                    bytes=position, rows=len(offsets), blank_lines=blanks,
                    index_sha256=_sha256(index_path))
        temporary = meta_path.with_suffix('.json.tmp')
        with open(temporary, 'w') as stream:
            json.dump(meta, stream, indent=2)
        temporary.replace(meta_path)
        return np.load(index_path, mmap_mode='r', allow_pickle=False), meta, index_path, False


def _split_from_name(path):
    parts = re.split(r'[^a-z0-9]+', Path(path).stem.lower())
    if 'test' in parts or 'quarantine' in parts:
        return 'test'
    if 'val' in parts or 'dev' in parts:
        return 'dev'
    if 'train' in parts:
        return 'train'
    return None


class SeeingDataset(Dataset):
    def __init__(self, manifest, root='data', split='train', image_hw=(640, 640), feature_hw=(160, 160),
                 illumination='original', augment=None, seed=0, center_shrink=.3,
                 index_cache_dir=None, image_fingerprints=None, verify_fingerprints=True,
                 require_fingerprint=False, protocol_provenance=None):
        if split not in ('train', 'dev'):
            raise ValueError('Only train/dev are accepted; test requires a frozen final-evaluation protocol')
        self.root, self.split = Path(root).resolve(), split
        requested = Path(manifest).resolve()
        if _split_from_name(requested) not in (None, split):
            raise ValueError(f'{requested.name} is not a {split} manifest; test/quarantine cannot enter training/dev')
        if not requested.is_file():
            prefix = 'MLT prerequisite is unavailable: ' if 'mlt' in requested.name.lower() else 'Manifest missing: '
            raise FileNotFoundError(prefix + str(requested))
        descriptor = None
        source = requested
        if requested.suffix.lower() != '.jsonl':
            if requested.stat().st_size > 5 * 1024 * 1024:
                raise ValueError('Expected a small JSON reference descriptor or a JSONL manifest')
            descriptor = _json(requested)
            if descriptor.get('status') == 'missing_prerequisite' or descriptor.get('usable_for_paper_recipe') is False:
                raise FileNotFoundError('MLT/full-data prerequisite is explicitly marked unavailable; no substitute is permitted')
            if not {'source_path', 'source_sha256', 'source_bytes'} <= descriptor.keys():
                raise ValueError('Reference descriptor must bind source_path, source_sha256 and source_bytes')
            source = Path(descriptor['source_path']).expanduser()
            if not source.is_absolute():
                source = requested.parent / source
            if not source.is_file():
                # Host absolute paths differ inside /workspace containers. This
                # bounded relocation is accepted ONLY after the full SHA check.
                source = self.root / 'annotations' / Path(descriptor['source_path']).name
            source = source.resolve()
            if not source.is_file():
                raise FileNotFoundError(f'Full referenced dataset is missing: {source}')
            if source.stat().st_size != descriptor['source_bytes']:
                raise ValueError('Referenced full-manifest byte count differs; do not substitute a subset')
        if _split_from_name(source) not in (None, split):
            raise ValueError('Referenced manifest split is incompatible with requested training/dev role')
        self.manifest = source
        self._offsets, index_meta, self._index_path, cache_hit = _index(
            source, index_cache_dir or Path(tempfile.gettempdir()) / 'seeread-v5-indices')
        if descriptor and index_meta['manifest_sha256'] != descriptor['source_sha256']:
            raise ValueError('Referenced full-manifest SHA256 differs; refusing substituted/modified data')
        self._manifest_stat = self._stat()
        protocol_path = Path(protocol_provenance) if protocol_provenance else requested.parent / 'provenance.json'
        if protocol_provenance and not protocol_path.is_file():
            raise FileNotFoundError(f'Protocol provenance missing: {protocol_path}')
        if protocol_path.is_file():
            protocol = _json(protocol_path)
            for name, entry in protocol.get('frozen_manifests', {}).items():
                same_content = entry.get('sha256') == index_meta['manifest_sha256']
                if same_content and ('test' in name.lower() or 'quarantine' in name.lower()
                                     or entry.get('role') == 'final_test_registration_only'):
                    raise ValueError('Manifest content is registered as held-out test/quarantine, even if renamed')
                if same_content and _split_from_name(name) not in (None, split):
                    raise ValueError('Manifest content belongs to a different registered train/dev role')
                if Path(entry.get('path', '')).name == requested.name and not same_content:
                    raise ValueError('Frozen V5 manifest hash differs from registered protocol')
        self.image_hw, self.feature_hw = tuple(map(int, image_hw)), tuple(map(int, feature_hw))
        if len(self.image_hw) != 2 or len(self.feature_hw) != 2 or min(*self.image_hw, *self.feature_hw) <= 0:
            raise ValueError('image_hw and actual feature_hw must contain positive H,W')
        if illumination not in ('original', 'fixed_synth'):
            raise ValueError(f'Unknown illumination policy: {illumination!r}')
        self.illumination, self.seed, self.center_shrink = illumination, int(seed), float(center_shrink)
        if self.seed < 0 or not 0 < self.center_shrink <= 1:
            raise ValueError('seed must be nonnegative and center_shrink must be in (0,1]')
        if augment is not None and not isinstance(augment, dict):
            augment = vars(augment)
        defaults = dict(scale_crop_p=0., scale_min=.75, scale_max=1.5,
                        flip_p=0., rotate_p=0., color_p=0., noise_p=0.,
                        rotate_degrees=10., color_strength=.2, noise_std=.02)
        if set(augment or {}) - defaults.keys():
            raise ValueError(f'Unknown augmentation fields: {set(augment) - defaults.keys()}')
        self.augment = {**defaults, **(augment or {})}
        for key, value in self.augment.items():
            if not np.isfinite(value) or value < 0 or (key.endswith('_p') and value > 1):
                raise ValueError(f'Invalid augmentation value: {key}')
        if self.augment['color_strength'] > 1 or self.augment['noise_std'] > 1 or self.augment['rotate_degrees'] > 180:
            raise ValueError('Augmentation magnitudes exceed declared limits')
        if not 0 < self.augment['scale_min'] <= self.augment['scale_max']:
            raise ValueError('Require 0 < scale_min <= scale_max')
        if split != 'train' and any(self.augment[k] for k in defaults if k.endswith('_p')):
            raise ValueError('Random training augmentation cannot be enabled on dev')
        self.verify_fingerprints, self.require_fingerprint = bool(verify_fingerprints), bool(require_fingerprint)
        if self.require_fingerprint and not self.verify_fingerprints:
            raise ValueError('require_fingerprint requires verification enabled')
        self.fingerprints, fingerprint_sources = {}, []
        fp_path = Path(image_fingerprints) if image_fingerprints else requested.parent / 'image_fingerprints.jsonl'
        if image_fingerprints and not fp_path.is_file():
            raise FileNotFoundError(f'Image fingerprint registry missing: {fp_path}')
        if fp_path.is_file():
            with open(fp_path) as stream:
                for line in stream:
                    if line.strip():
                        row = json.loads(line)
                        self.fingerprints[row.get('path', row.get('image'))] = row
            fingerprint_sources.append(dict(path=str(fp_path.resolve()), sha256=_sha256(fp_path)))
        for row in (descriptor or {}).get('image_sample') or []:
            self.fingerprints.setdefault(row['image'], row)
        self.epoch, self._reader, self._reader_pid = 0, None, None
        self.provenance = dict(manifest=str(source), requested_manifest=str(requested),
                               manifest_sha256=index_meta['manifest_sha256'], manifest_bytes=index_meta['bytes'],
                               rows=len(self), split=split, index_format=INDEX_FORMAT, index=str(self._index_path),
                               index_cache_hit=cache_hit, image_hw=list(self.image_hw), feature_hw=list(self.feature_hw),
                               resize='rectangular RGB bilinear stretch; polygons use independent x/y scales',
                               exif_orientation='ignored; encoded pixel matrix matches header coordinates',
                               illumination=illumination, seed=self.seed, center_shrink=self.center_shrink,
                               augmentation=self.augment.copy(), augmentation_probabilities_from_paper=False,
                               augmentation_assumption='probabilities unspecified in paper; defaults all zero; configured values are assumptions',
                               scale_crop_policy='isotropic affine after base resize; uniform crop/pad offset; '
                                                 'black padding; original polygons retained for clipped raster targets; '
                                                 'scale range and policy are unpublished-detail assumptions',
                               verify_fingerprints=self.verify_fingerprints, require_fingerprint=self.require_fingerprint,
                               known_image_fingerprints=len(self.fingerprints), fingerprint_sources=fingerprint_sources,
                               full_image_hash_coverage_claimed=False,
                               reference_sha256=_sha256(requested) if descriptor else None,
                               reference_declared_source=(descriptor or {}).get('source_path'))

    def _stat(self):
        s = self.manifest.stat()
        return s.st_size, s.st_mtime_ns, s.st_ino

    def __len__(self):
        return len(self._offsets)

    def set_epoch(self, epoch):
        """Call before creating workers; persistent samplers can yield (epoch,index)."""
        if int(epoch) < 0:
            raise ValueError('epoch must be nonnegative')
        self.epoch = int(epoch)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_reader'] = state['_offsets'] = state['_reader_pid'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._offsets = np.load(self._index_path, mmap_mode='r', allow_pickle=False)

    def _row(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._stat() != self._manifest_stat:
            raise RuntimeError('Manifest changed after dataset construction; rebuild the SHA-bound index')
        if self._reader is None or self._reader_pid != os.getpid():
            if self._reader is not None:
                self._reader.close()
            self._reader, self._reader_pid = open(self.manifest, 'rb'), os.getpid()
        self._reader.seek(int(self._offsets[index]))
        row = json.loads(self._reader.readline())
        declared_split = row.get('split', row.get('meta', {}).get('split'))
        if declared_split is not None and ('dev' if declared_split == 'val' else declared_split) != self.split:
            raise ValueError('Row split disagrees with dataset role; test data cannot train/select a model')
        return row

    def _path(self, image):
        relative = Path(image)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Image path must be relative to the declared data root')
        if any(_split_from_name(part) == 'test' for part in relative.parts[:-1]):
            raise ValueError('Held-out test image path cannot enter train/dev')
        path = (self.root / relative).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError('Image symlink escapes the declared data root') from error
        return path

    def __getitem__(self, index):
        epoch = self.epoch
        if isinstance(index, tuple):
            epoch, index = map(int, index)
        index = int(index)
        row = self._row(index)
        image_path = self._path(row['image'])
        fingerprint = self.fingerprints.get(row['image'])
        row_digest = row.get('sha256')
        if isinstance(row_digest, dict):
            row_digest = row_digest.get('image')
        expected = row_digest or (fingerprint or {}).get('sha256')
        if self.require_fingerprint and not expected:
            raise ValueError(f'No image fingerprint available: {row["image"]}')
        if self.verify_fingerprints and expected and _sha256(image_path) != expected:
            raise ValueError(f'Image fingerprint mismatch: {row["image"]}')
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_hw = tuple(image.shape[:2])
        if fingerprint and all(k in fingerprint for k in ('height', 'width')):
            if original_hw != (fingerprint['height'], fingerprint['width']):
                raise ValueError('Image dimensions differ from registered fingerprint')
        illumination_meta = {'policy': self.illumination}
        if self.illumination == 'fixed_synth':
            image, dark_meta = darken(image, np.random.default_rng((self.seed, index, 0xD4A)), mode='physical')
            illumination_meta.update(dark_meta)
        ih, iw = self.image_hw
        image = cv2.resize(image, (iw, ih), interpolation=cv2.INTER_LINEAR)
        original_instances = copy.deepcopy(row['instances'])
        scale = np.array([iw / original_hw[1], ih / original_hw[0]], np.float32)
        instances = [{**ins, 'poly': np.asarray(ins['poly'], np.float32) * scale} for ins in original_instances]
        rng = np.random.default_rng((self.seed, epoch, index, 0xA09))
        applied = {}
        if self.augment['scale_crop_p'] > 0 and rng.random() < self.augment['scale_crop_p']:
            factor = rng.uniform(self.augment['scale_min'], self.augment['scale_max'])
            dx, dy = (iw - 1) * (1 - factor), (ih - 1) * (1 - factor)
            matrix = np.array([[factor, 0, rng.uniform(min(0, dx), max(0, dx))],
                               [0, factor, rng.uniform(min(0, dy), max(0, dy))]], np.float32)
            image = cv2.warpAffine(image, matrix, (iw, ih), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT)
            for instance in instances:
                instance['poly'] = np.c_[instance['poly'], np.ones(len(instance['poly']))] @ matrix.T
            applied['scale_crop_affine'] = matrix.tolist()
        if rng.random() < self.augment['flip_p']:
            image = image[:, ::-1].copy()
            for instance in instances:
                instance['poly'][:, 0] = iw - 1 - instance['poly'][:, 0]
            applied['horizontal_flip'] = True
        if rng.random() < self.augment['rotate_p']:
            angle = rng.uniform(-self.augment['rotate_degrees'], self.augment['rotate_degrees'])
            matrix = cv2.getRotationMatrix2D(((iw - 1) / 2, (ih - 1) / 2), angle, 1.)
            image = cv2.warpAffine(image, matrix, (iw, ih), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            for instance in instances:
                instance['poly'] = np.c_[instance['poly'], np.ones(len(instance['poly']))] @ matrix.T
            applied['rotation_degrees'] = float(angle)
        if rng.random() < self.augment['color_p']:
            strength = self.augment['color_strength']
            gains = rng.uniform(1 - strength, 1 + strength, 3)
            image = np.clip(image.astype(np.float32) * gains, 0, 255).astype(np.uint8)
            applied['rgb_gains'] = gains.tolist()
        if rng.random() < self.augment['noise_p']:
            noise = rng.normal(0, 255 * self.augment['noise_std'], image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            applied['noise_std'] = self.augment['noise_std']
        target = targets_from_polygons(instances, self.image_hw, self.feature_hw, self.center_shrink)
        return dict(image=torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255,
                    targets={k: torch.from_numpy(v) for k, v in target.items()},
                    original_instances=original_instances,
                    transformed_instances=[{**ins, 'poly': ins['poly'].tolist()} for ins in instances],
                    original_hw=original_hw, image_hw=self.image_hw, feature_hw=self.feature_hw,
                    image_path=row['image'], index=index, epoch=epoch,
                    metadata=row.get('meta', {}), illumination=illumination_meta, augmentation=applied,
                    image_fingerprint_verified=bool(self.verify_fingerprints and expected))

    def __del__(self):
        reader = getattr(self, '_reader', None)
        if reader is not None:
            reader.close()


def collate_seeing(batch):
    if not batch:
        raise ValueError('Cannot collate an empty detection batch')
    output = dict(image=torch.stack([row['image'] for row in batch]),
                  targets={k: torch.stack([row['targets'][k] for row in batch]) for k in batch[0]['targets']})
    for key in batch[0]:
        if key not in output:
            output[key] = [row[key] for row in batch]
    return output
