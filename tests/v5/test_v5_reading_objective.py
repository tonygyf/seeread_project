"""Published objective weights and enhancement/recognition gradient coupling."""
import os
import unittest
from pathlib import Path

import torch
from torch import nn

from seeread.v5.reading_objective import ReadingJointObjective, PaperReadingPipeline, sobel_magnitude
from seeread.v5.enhancers import SelfDACEEnhancer, _load_model_source


class SobelTests(unittest.TestCase):
    def test_constant_and_identical_images_have_zero_edge_difference(self):
        image = torch.full((2, 3, 12, 16), .2, requires_grad=True)
        value = sobel_magnitude(image)
        self.assertTrue(torch.allclose(value, torch.full_like(value, 1e-6)))
        loss = (value - sobel_magnitude(image.detach())).square().mean()
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(image.grad).all())


@unittest.skipUnless(os.getenv('SELF_DACE_SOURCE_ROOT'), 'official Self-DACE checkout required')
class ObjectiveTests(unittest.TestCase):
    def test_lle_matches_upstream_stage1_weighted_loss(self):
        torch.set_num_threads(1)
        root = Path(os.environ['SELF_DACE_SOURCE_ROOT'])
        objective = ReadingJointObjective(root)
        upstream = _load_model_source(root / 'codes_SelfDACE/old_version/stage1/Myloss.py')
        torch.manual_seed(3)
        low, enhanced = torch.rand(2, 3, 16, 24), torch.rand(2, 3, 16, 24)
        alpha, beta = torch.rand(2, 21, 16, 24), torch.rand(2, 21, 16, 24)
        result = objective.enhancement_terms(low, enhanced, alpha, beta)
        expected = (1000 * upstream.L_localcolor()(enhanced, low).mean() +
                    1500 * upstream.L_color()(enhanced).mean() +
                    5 * upstream.L_exp(4, .8)(enhanced, low).mean() +
                    1000 * upstream.L_SMO()(alpha) + 5000 * upstream.L_SMO()(beta))
        self.assertTrue(torch.equal(result['lle'], expected))
        self.assertTrue(torch.equal(result['llie'], expected + 1e5 * result['edge']))

    def test_total_loss_routes_gradients_into_both_modules(self):
        torch.set_num_threads(1)
        root = os.environ['SELF_DACE_SOURCE_ROOT']
        class Recognition(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(1.))
            def forward_loss(self, images, texts):
                return (self.scale * images.mean() - .7).square()
        model = PaperReadingPipeline(SelfDACEEnhancer(root, stage=1, freeze=False),
                                     Recognition(), ReadingJointObjective(root))
        value = model.forward_loss(torch.rand(2, 3, 16, 24) * .2, ['a', 'b'])
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertGreater(float(model.recognizer.scale.grad.abs()), 0.)
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                            for p in model.enhancer.parameters()))
        terms = model.last_loss_components
        self.assertAlmostEqual(terms['total'], terms['llie'] + 100 * terms['ocr'], delta=1e-2)


if __name__ == '__main__':
    unittest.main()
