import unittest
import copy

from tools.data.prepare_v5_connector_reading import select_train
from tools.v5_extend_reading import extension_gate


class ConnectorSplitTests(unittest.TestCase):
    def test_corrected_annotations_use_only_frozen_training_scenes(self):
        candidate = [dict(image='train', text='corrected'), dict(image='dev'), dict(image='test')]
        result = select_train(candidate, [dict(image='train')], [dict(image='dev')], [dict(image='test')])
        self.assertEqual(result, [candidate[0]])
        for heldout in ('dev', 'test'):
            with self.assertRaisesRegex(ValueError, 'overlap'):
                select_train(candidate, [dict(image=heldout)], [dict(image='dev')], [dict(image='test')])

    def test_more_epochs_require_cer_gain_without_dark_spotting_regression(self):
        def report(cer, dark_f=.1, bright_f=.2):
            return dict(status='complete', data_hashes={'dev': 'fixed'}, weights={'det.pt': 'same'},
                        source_hashes={name: 'same' for name in ('seeread/v5/spotting.py',
                                      'seeread/models/rectify.py', 'seeread/v5/seeing_decode.py')},
                        config=dict(weights={'detector': 'det.pt'}, crop={'pad': 1.1},
                                    cases=['boundary_when_supported'], enhancement='full'),
                        results={key: dict(complete=True, cases={'boundary_when_supported':
                            dict(gt_crop_raw={'cer': cer}, spotting={'e2e_F': value})})
                            for key, value in [('bright', bright_f), ('dark', dark_f)]})
        base = report(.65)
        self.assertTrue(extension_gate(base, report(.63))['passed'])
        self.assertFalse(extension_gate(base, report(.645))['passed'])
        self.assertFalse(extension_gate(base, report(.63, dark_f=.09))['passed'])
        self.assertFalse(extension_gate(base, report(.63, bright_f=.18))['passed'])
        mismatch = copy.deepcopy(base)
        mismatch['weights']['det.pt'] = 'different'
        with self.assertRaisesRegex(ValueError, 'identical detector'):
            extension_gate(base, mismatch)


if __name__ == '__main__':
    unittest.main()
