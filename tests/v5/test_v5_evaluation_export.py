"""CPU fixtures exercise GT equality, complete exports and protocol coordinates."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from scipy.io import savemat

from seeread.v5.evaluation_export import export_predictions, sha256


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.gt = self.root / 'gt'; self.gt.mkdir()
        self.data = self.root / 'data'; self.data.mkdir()
        self.manifest = self.root / 'test.jsonl'
        self.predictions = self.root / 'predictions.jsonl'
        self.poly = [[10, 20], [60, 20], [60, 45], [10, 45]]
        self.ignore_poly = [[70, 10], [90, 10], [90, 40], [70, 40]]

    def tearDown(self):
        self.temp.cleanup()

    def prepare_tt(self):
        rows, records = [], []
        for index in (1, 2):
            name = f'img{index}.jpg'
            Image.new('RGB', (120, 70)).save(self.data / name)
            instances = [dict(poly=self.poly, ignore=False), dict(poly=self.ignore_poly, ignore=True)]
            rows.append(dict(image=name, instances=instances))
            records.append(dict(image=name, original_hw=[70, 120], predictions=[] if index == 2 else
                                [dict(poly=[[10.7,20.9],[60.7,20.9],[60.7,45.9],[10.7,45.9]], score=.3),
                                 dict(poly=self.ignore_poly, score=.9)]))
            table = np.empty((2, 6), dtype=object)
            for i, item in enumerate(instances):
                p = np.array(item['poly'])
                table[i] = [np.array([]), p[:, 0][None], np.array([]), p[:, 1][None], 'word', '#' if item['ignore'] else 'h']
            savemat(self.gt / f'poly_gt_img{index}.mat', dict(polygt=table))
        self.write(rows, records)
        return rows, records

    def write(self, rows, records):
        self.manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        self.predictions.write_text(''.join(json.dumps(r) + '\n' for r in records))

    def export(self, name='output', **kw):
        return export_predictions(self.manifest, self.predictions, self.root / name,
                                  data_root=self.data, gt_root=self.gt,
                                  protocols=kw.pop('protocols', ['totaltext_deteval', 'totaltext_pascal']),
                                  quantization=kw.pop('quantization', 'truncate'), **kw)

    def test_tt_independent_yx_files_empty_image_and_hashes(self):
        self.prepare_tt()
        audit = self.export()
        for protocol in ('totaltext_deteval', 'totaltext_pascal'):
            directory = self.root / 'output' / protocol
            self.assertEqual((directory / 'img2.txt').read_bytes(), b'')
            self.assertEqual((directory / 'img1.txt').read_text().splitlines()[0], '20,10,20,60,45,60,45,10')
            self.assertEqual(len((directory / 'img1.txt').read_text().splitlines()), 2) # ignore predictions retained
        for name, digest in audit['output_sha256'].items():
            self.assertEqual(sha256(self.root / 'output' / name), digest)
        self.assertEqual(audit['manifest']['sha256'], sha256(self.manifest))
        self.assertFalse(audit['evaluation_executed'])
        self.assertEqual(audit['protocols']['totaltext_deteval']['commit'], '59538b936fc7a55025987960270267d8d081e72e')
        canonical = json.loads((self.root/'output/canonical_predictions.jsonl').read_text().splitlines()[0])
        self.assertEqual([p['score'] for p in canonical['predictions']], [.3,.9])
        self.assertEqual(canonical['predictions'][0]['poly'][0], [10.7,20.9])
        with self.assertRaises(FileExistsError): self.export()

    def test_missing_gt_coverage_duplicate_names_and_wrong_frame_rejected(self):
        rows, records = self.prepare_tt()
        (self.gt / 'poly_gt_img2.mat').unlink()
        with self.assertRaises(FileNotFoundError): self.export()
        rows, records = self.prepare_tt()
        self.write(rows, records[:1])
        with self.assertRaisesRegex(ValueError, 'every manifest'): self.export()
        self.write(rows, records + [records[0]])
        with self.assertRaisesRegex(ValueError, 'collision'): self.export()
        records[0]['original_hw'] = [120, 70]
        self.write(rows, records)
        with self.assertRaisesRegex(ValueError, 'dimensions'): self.export()
        self.assertFalse((self.root / 'output').exists())

    def test_gt_ignore_order_geometry_and_quantization_fail_closed(self):
        rows, records = self.prepare_tt()
        rows[0]['instances'][0]['poly'] = self.poly[::-1] # exact reversed ring allowed
        self.write(rows, records)
        self.export(name='reversed')
        rows[0]['instances'][1]['ignore'] = False
        self.write(rows, records)
        with self.assertRaisesRegex(ValueError, 'ignore/order'): self.export()
        rows, records = self.prepare_tt()
        rows[0]['instances'].reverse(); self.write(rows, records)
        with self.assertRaisesRegex(ValueError, 'ignore/order'): self.export()
        rows, records = self.prepare_tt()
        with self.assertRaisesRegex(ValueError, 'quantization'): self.export(quantization='none')
        records[0]['predictions'][0]['score'] = float('nan'); self.write(rows, records)
        with self.assertRaisesRegex(ValueError, 'Nonfinite'): self.export()
        rows, records = self.prepare_tt()
        records[0]['predictions'][0]['poly'] = [[1.1,1.1],[1.2,1.1],[1.2,1.2]]
        self.write(rows, records)
        with self.assertRaisesRegex(ValueError, 'Invalid/zero-area'): self.export()
        self.assertFalse((self.root / 'output').exists())

    def test_exact_author_gt_may_be_single_point_or_self_intersecting(self):
        # Actual author MAT records: img659 instance15 and img557 instance8.
        # These GT coordinates are evidence to compare, not predictions to repair.
        cases = [([[39, 292]], True, '#'),
                 ([[2026,1290],[2175,1278],[2183,1326],[2014,1358],
                   [2183,1322],[2006,1350]], False, 'h')]
        for index, (poly, ignore, marker) in enumerate(cases):
            with self.subTest(marker=marker):
                rows, records = self.prepare_tt()
                rows[0]['instances'] = [dict(poly=poly, ignore=ignore)]
                Image.new('RGB', (2300, 1500)).save(self.data / 'img1.jpg')
                records[0]['original_hw'] = [1500, 2300]
                table = np.empty((1, 6), dtype=object)
                p = np.asarray(poly)
                table[0] = [np.array([]), p[:, 0][None], np.array([]),
                            p[:, 1][None], '#' if ignore else 'evo', marker]
                original = self.gt / 'poly_gt_img1.mat'
                savemat(original, dict(polygt=table))
                original_bytes = original.read_bytes()
                self.write(rows, records)
                audit = self.export(name=f'author_gt_{index}')
                self.assertEqual(original.read_bytes(), original_bytes)
                self.assertEqual(audit['original_gt_sha256'][str(original)], sha256(original))
                canonical = json.loads((self.root / f'author_gt_{index}' /
                                        'canonical_predictions.jsonl').read_text().splitlines()[0])
                self.assertEqual(canonical['predictions'], records[0]['predictions'])
                # Malformed source geometry does not relax exact-coordinate/ignore equality.
                rows[0]['instances'][0]['ignore'] = not ignore
                self.write(rows, records)
                with self.assertRaisesRegex(ValueError, 'ignore/order'):
                    self.export(name=f'wrong_ignore_{index}')
                rows[0]['instances'][0]['ignore'] = ignore
                rows[0]['instances'][0]['poly'] = [[x + 1, y] for x, y in poly]
                self.write(rows, records)
                with self.assertRaisesRegex(ValueError, 'ignore/order'):
                    self.export(name=f'wrong_coordinate_{index}')
                rows[0]['instances'][0]['poly'] = poly
                records[0]['predictions'] = [dict(poly=poly, score=.3)]
                self.write(rows, records)
                with self.assertRaisesRegex(ValueError, 'three finite|Invalid/zero-area'):
                    self.export(name=f'invalid_prediction_{index}')

    def test_ctw_raw_offsets_explicit_author_order_and_float_export(self):
        rows, records = [], []
        for index in (1001, 1002):
            name = f'{index}.jpg'; Image.new('RGB',(120,70)).save(self.data/name)
            # 14 vertices around a rectangle, exactly original author offsets.
            p = [[10 + 5*k,20] for k in range(7)] + [[40-5*k,45] for k in range(7)]
            values = [10,20,40,45] + [v for x,y in p for v in (x-10,y-20)]
            (self.gt/f'{index}.txt').write_text(','.join(map(str,values))+'\n')
            rows.append(dict(image=name, instances=[dict(poly=p[::-1],ignore=False)]))
            records.append(dict(image=name,original_hw=[70,120],predictions=[] if index==1002 else
                                [dict(poly=[[10.5,20],[60.5,20],[60.5,45],[10.5,45]],score=.2)]))
        # Manifest order need not be the evaluator order.
        self.write(rows[::-1], records[::-1])
        imageset=self.root/'test.txt'; imageset.write_text('1001.jpg\n1002.jpg\n')
        annotations=self.root/'test_label_curve.txt'; annotations.write_text('1001.txt\n1002.txt\n')
        kwargs=dict(protocols=['ctw1500_voc'],quantization='none',ctw_imageset=imageset,ctw_annotations=annotations)
        audit=self.export(**kwargs)
        self.assertEqual((self.root/'output/ctw1500_voc/1001.txt').read_text(), '10.5,20,60.5,20,60.5,45,10.5,45\n')
        self.assertEqual((self.root/'output/ctw1500_voc/1002.txt').read_bytes(),b'')
        self.assertTrue((self.root/'output/ctw1500_voc_detections_text0.5.txt').read_text().startswith('0 0.999 10.5 20 '))
        self.assertEqual([m['evaluator_id'] for m in audit['mapping']],['0','1'])
        imageset.write_text('1002.jpg\n1001.jpg\n')
        with self.assertRaisesRegex(ValueError,'order/id'): self.export(name='bad',**kwargs)


if __name__ == '__main__': unittest.main()
