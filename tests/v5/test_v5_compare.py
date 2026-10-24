"""Independent saved-prediction comparisons must retain their paired unit."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('v5_compare', Path(__file__).parents[2] / 'tools/v5_compare_reading.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ComparisonTests(unittest.TestCase):
    def row(self, key, text, prediction):
        return dict(id=key, text=text, prediction=prediction, image=key+'.png',
                    source='test', source_group=key, split='dev')

    def test_known_paired_difference(self):
        a = [self.row('a', 'CAT', 'CAR'), self.row('b', 'A', '')]
        b = [self.row('b', 'A', 'A'), self.row('a', 'CAT', 'CAT')]
        r = module.compare(a, b, iterations=200)
        self.assertEqual(r['policies']['raw']['cer_difference'], -.5)
        self.assertLess(r['policies']['raw']['paired_bootstrap_95_percentile_interval'][1], 0)

    def test_unpaired_or_changed_reference_rejected(self):
        a = [self.row('a', 'CAT', 'CAT')]
        for b in ([self.row('b', 'CAT', 'CAT')], [self.row('a', 'DOG', 'DOG')], a+a):
            with self.assertRaises(ValueError):
                module.compare(a, b)


if __name__ == '__main__':
    unittest.main()
