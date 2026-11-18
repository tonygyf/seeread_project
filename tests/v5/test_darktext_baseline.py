"""Adapter contract tests using a synthetic detector and no dataset."""
import unittest

import numpy as np
import torch

from tools.darktext_baseline import infer


class FakeDetector:
    def detect(self, image, _):
        assert tuple(image.shape) == (1, 3, 64, 64)
        assert not image.requires_grad
        polygon = np.array([[8, 16], [40, 16], [40, 32], [8, 32]])
        return [[polygon, polygon]], {}


class DarkTextAdapterTest(unittest.TestCase):
    def test_polygon_returns_to_original_non_square_coordinates(self):
        result = infer(FakeDetector(), np.zeros((128, 256, 3), np.uint8), torch.device('cpu'), size=64)
        self.assertEqual(result, [{'poly': [[32., 32.], [160., 32.], [160., 64.], [32., 64.]],
                                   'score': None}])


if __name__ == '__main__':
    unittest.main()
