import tempfile
import json
import subprocess
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

from seeread.v5.reading import sha256
from seeread.v5.lated_annotations import write_jsonl
from seeread.v5.scene_str import audit_splits, crop_views, export_reviewed, score_catalog, quality_hint, pair_training


class SceneSTRTests(unittest.TestCase):
    def test_reviewed_sidecar_to_two_crop_manifests_and_pairing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); package = root/'package'; package.mkdir()
            tasks, annotations, sources = [], [], {}
            for split, value in [('train', 80), ('dev', 160)]:
                image = split+'.png'; cv2.imwrite(str(root/image), np.full((30, 60, 3), value, np.uint8))
                poly = [[5, 5], [50, 5], [50, 20], [5, 20]]
                manifest = root/(split+'.jsonl')
                write_jsonl(manifest, [dict(image=image, instances=[dict(id=split, poly=poly, text='###', ignore=False)])])
                sources[split] = dict(path=str(manifest), sha256=sha256(manifest))
                tasks.append(dict(id=split, image=image, split=split, poly=poly,
                                  image_sha256=sha256(root/image), pilot=True))
                annotations.append(dict(id=split, image=image, split=split, status='reviewed',
                                        language='latin', reviewer='unit-test', text='HELLO'))
            write_jsonl(package/'tasks.jsonl', tasks)
            (package/'protocol.json').write_text(json.dumps(dict(schema='lated_transcription_v1',
                data_root=str(root), sources=sources, tasks_sha256=sha256(package/'tasks.jsonl'))))
            sidecar = root/'sidecar.json'
            sidecar.write_text(json.dumps(dict(schema='lated_transcription_sidecar_v1',
                protocol_sha256=sha256(package/'protocol.json'), annotations=annotations)))
            tool = str(Path(__file__).resolve().parents[2]/'tools/v5_scene_str.py')
            for arm, method in [('a', 'axis_aligned'), ('b', 'boundary_when_supported')]:
                subprocess.run([sys.executable, tool, 'export', '--package', str(package), '--sidecar', str(sidecar),
                                '--pilot', '--method', method, '--output', str(root/arm)], check=True, capture_output=True)
            subprocess.run([sys.executable, tool, 'pair-train', '--a', str(root/'a'), '--b', str(root/'b'),
                            '--output', str(root/'paired')], check=True, capture_output=True)
            for arm in ('a', 'b'):
                row = json.loads((root/'paired'/arm/'train.jsonl').read_text())
                self.assertEqual(row['text'], 'HELLO')
                self.assertEqual(row['split'], 'train')
                self.assertTrue((root/row['image']).exists())

    def test_test_difficulty_is_not_changed_by_crop_failures(self):
        catalog = [dict(id='a', text='AB', eligible=True, crop_error=None),
                   dict(id='b', text='CD', eligible=True, crop_error='invalid_polygon'),
                   dict(id='c', text='', eligible=False, crop_error=None)]
        result = score_catalog(catalog, [dict(id='a', prediction='AB')])
        self.assertEqual(result['raw']['cer'], .5)
        self.assertEqual(result['raw']['word_accuracy'], .5)
        self.assertEqual(result['total_source_instances'], 3)
        with self.assertRaises(ValueError):
            score_catalog(catalog, [])

    def test_native_and_rectified_crops_keep_difficulty_metadata(self):
        image = np.full((80, 180, 3), 80, np.uint8)
        polygon = [[20, 20], [150, 30], [150, 55], [20, 45]]
        native, q, error = crop_views(image, polygon)
        self.assertIsNone(error)
        self.assertEqual(native.shape, (36, 131, 3))
        result, other, error = crop_views(image, polygon, 'boundary_when_supported')
        self.assertIsNone(error)
        self.assertEqual(result.shape, (384, 384, 3))
        self.assertEqual(q['native_hw'], other['native_hw'])
        self.assertFalse(q['quality_is_readability_label'])

    def test_quality_hints_do_not_delete_difficult_crops(self):
        flat = quality_hint(np.zeros((5, 5, 3), np.uint8))
        self.assertTrue(flat['needs_extreme_review'])
        self.assertFalse(flat['automatic_exclusion'])
        text = np.zeros((5, 5, 3), np.uint8); text[1:3] = 255
        self.assertFalse(quality_hint(text)['needs_extreme_review'])

    def test_crop_ablation_uses_identical_training_ids(self):
        row = dict(id='a', text='WORD', source_group='scene1', source='LATeD', split='train')
        a, b = pair_training([row, dict(row, id='b')], [dict(row, image='othercrop')])
        self.assertEqual([x['id'] for x in a], ['a'])
        self.assertEqual([x['id'] for x in b], ['a'])
        with self.assertRaises(ValueError):
            pair_training([row], [dict(row, text='WRONG')])

    def test_invalid_crops_are_explicit(self):
        image = np.zeros((20, 20, 3), np.uint8)
        self.assertEqual(crop_views(image, [[-1, 1], [10, 1], [10, 10]])[2], 'polygon_outside_image')
        self.assertEqual(crop_views(image, [[1, 1], [2, 2], [3, 3]])[2], 'degenerate_polygon')

    def test_scene_byte_duplicates_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'a.png').write_bytes(b'image'); (root/'b.png').write_bytes(b'image')
            with self.assertRaisesRegex(ValueError, 'crosses splits'):
                audit_splits(root, {'train':[dict(image='a.png',instances=[])],
                                    'dev':[dict(image='b.png',instances=[])]})

    def test_reviewed_labels_required_and_export_preserves_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = root/'scene.png'
            cv2.imwrite(str(path), np.full((20, 50, 3), 128, np.uint8))
            before = path.read_bytes()
            gt = dict(id='line0', poly=[[1, 1], [40, 1], [40, 15], [1, 15]],
                      text='Hello', lang='latin', ocr_eligible=True, transcription_status='pending')
            rows = [dict(image='scene.png', instances=[gt])]
            with self.assertRaises(ValueError):
                export_reviewed(rows, root, root/'crops', 'train', 'axis_aligned')
            gt['transcription_status'] = 'reviewed'
            words, catalog = export_reviewed(rows, root, root/'crops', 'train', 'axis_aligned')
            self.assertEqual(len(words), 1); self.assertEqual(len(catalog), 1)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(words[0]['source_group'], 'scene.png')


if __name__ == '__main__':
    unittest.main()
