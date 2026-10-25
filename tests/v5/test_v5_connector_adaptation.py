import unittest

from tools.data.prepare_v5_connector_reading import select_train


class ConnectorSplitTests(unittest.TestCase):
    def test_corrected_annotations_use_only_frozen_training_scenes(self):
        candidate = [dict(image='train', text='corrected'), dict(image='dev'), dict(image='test')]
        result = select_train(candidate, [dict(image='train')], [dict(image='dev')], [dict(image='test')])
        self.assertEqual(result, [candidate[0]])
        for heldout in ('dev', 'test'):
            with self.assertRaisesRegex(ValueError, 'overlap'):
                select_train(candidate, [dict(image=heldout)], [dict(image='dev')], [dict(image='test')])


if __name__ == '__main__':
    unittest.main()
