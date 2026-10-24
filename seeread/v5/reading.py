"""Word-crop data and metrics for independent V5 reading experiments."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from seeread.metrics.spotting import edit_distance


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def normalize(text, policy):
    if policy == 'raw':
        return text
    if policy == 'lower_alnum':
        return ''.join(c for c in text if c.isalnum()).lower()
    raise ValueError(f'Unknown text normalization: {policy!r}')


def metrics(records, policy='raw'):
    normalize('', policy)  # Reject typos even for an interrupted/empty evaluation.
    edits = chars = correct = empty = 0
    for row in records:
        ref, pred = normalize(row['text'], policy), normalize(row['prediction'], policy)
        edits += edit_distance(pred, ref)
        chars += len(ref)
        correct += pred == ref
        empty += not ref
    return dict(normalization=policy, character_unit='Unicode code points', samples=len(records),
                reference_characters=chars, edit_operations=edits, correct_words=correct,
                empty_references=empty, cer=edits / chars if chars else None,
                word_accuracy=correct / len(records) if records else None)


class WordCropDataset(Dataset):
    allowed_splits = ('train', 'dev')

    def __init__(self, manifest, root, split, input_hw=(384, 384), verify_hashes=True):
        if split not in self.allowed_splits:
            raise ValueError('Reading model selection accepts train/dev only; test requires a separate final protocol')
        self.manifest, self.root = Path(manifest), Path(root)
        self.input_hw, self.verify_hashes = tuple(input_hw), verify_hashes
        if len(self.input_hw) != 2 or any(int(n) <= 0 for n in self.input_hw):
            raise ValueError('input_hw must contain positive height and width')
        self.rows = [json.loads(line) for line in self.manifest.read_text().splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f'Empty {split} manifest')
        ids = set()
        for row in self.rows:
            missing = {'id', 'image', 'text', 'source', 'source_group', 'split', 'dataset', 'sha256'} - row.keys()
            if missing:
                raise ValueError(f'Manifest missing fields: {sorted(missing)}')
            if row['id'] in ids:
                raise ValueError(f'Duplicate word id: {row["id"]}')
            ids.add(row['id'])
            if row['split'] != split:
                raise ValueError(f'Expected {split} row, received {row["split"]!r}')
            if not isinstance(row['text'], str) or not row['text']:
                raise ValueError('Transcription must be a nonempty string')
            self.image_path(row)
            if not isinstance(row['sha256'], dict) or not row['sha256'].get('image'):
                raise ValueError('sha256 must contain image digest and optional lit digest')
        self.provenance = dict(path=str(self.manifest.resolve()), sha256=sha256(self.manifest),
                               split=split, samples=len(self.rows), input_hw=list(self.input_hw),
                               sources=sorted({r['source'] for r in self.rows}),
                               scene_groups_verified=all(r.get('scene_group_verified', False) for r in self.rows))

    def image_path(self, row):
        relative = Path(row['image'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('image must be relative to data.root without parent traversal')
        return self.root / relative

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.image_path(row)
        if self.verify_hashes and sha256(path) != row['sha256']['image']:
            raise ValueError(f'Image digest mismatch: {row["id"]}')
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (int(self.input_hw[1]), int(self.input_hw[0])), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255
        return tensor, row


def collate_words(batch):
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch]


def check_disjoint(train, dev):
    for key in ('id', 'source_group', 'image'):
        overlap = {r[key] for r in train.rows} & {r[key] for r in dev.rows}
        if overlap:
            raise ValueError(f'train/dev overlap in {key}: {next(iter(overlap))!r}')
    def hashes(dataset):
        return {h for r in dataset.rows for h in r['sha256'].values() if h}
    if hashes(train) & hashes(dev):
        raise ValueError('train/dev share identical image/lit content')
