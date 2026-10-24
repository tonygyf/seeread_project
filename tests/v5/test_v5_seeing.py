"""SCM labels must never affect the main prediction, even during training."""
import unittest
import torch
from torch import nn

from seeread.v5.seeing import PerceptualModulation, SeeingDetector


class FakeDSConv(nn.Conv2d):
    def __init__(self, cin, cout, **kwargs):
        super().__init__(cin, cout, 3, padding=1)


class SeeingTests(unittest.TestCase):
    def test_pm_convolution_stays_finite_before_batch_normalization(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        model = PerceptualModulation(4, 4, source_root='', dsconv_factory=FakeDSConv).train()
        with torch.no_grad():
            model.conv[0].weight.fill_(50000.)
        image = torch.rand(2, 4, 16, 16, requires_grad=True)
        reference = model(image).detach()
        with torch.autocast('cpu', dtype=torch.float16):
            prediction = model(image)
        self.assertTrue(torch.isfinite(prediction).all())
        torch.testing.assert_close(prediction.float(), reference, rtol=.05, atol=.05)
        prediction.float().square().mean().backward()
        self.assertTrue(torch.isfinite(model.conv[0].weight.grad).all())
        self.assertTrue(torch.isfinite(image.grad).all())

    def test_classification_heads_stay_finite_when_convolution_exceeds_fp16_range(self):
        torch.set_num_threads(1)
        for name in ('center', 'text'):
            with self.subTest(head=name):
                torch.manual_seed(4)
                model = SeeingDetector(channels=4, pretrained=False, dsconv_factory=FakeDSConv).train()
                # Real SynthText failures overflowed each classification
                # convolution before BN could normalize finite FP32 values.
                head = getattr(model, name + '_head')
                with torch.no_grad():
                    head[0][0].weight.fill_(50000.)
                image = torch.rand(2, 3, 64, 96)
                reference = model(image)[name].detach()
                with torch.autocast('cpu', dtype=torch.float16):
                    prediction = model(image)[name]
                self.assertTrue(torch.isfinite(prediction).all())
                torch.testing.assert_close(prediction.float(), reference, rtol=.05, atol=.05)
                prediction.float().square().mean().backward()
                self.assertTrue(torch.isfinite(head[0][0].weight.grad).all())
                self.assertTrue(torch.isfinite(model.pm[-1].output_projection.weight.grad).all())

    def test_attention_reduction_stays_finite_under_fp16_autocast(self):
        torch.set_num_threads(1)
        model = PerceptualModulation(4, 4, source_root='', dsconv_factory=FakeDSConv).eval()
        with torch.no_grad():
            for projection in (model.q, model.k):
                projection.weight.zero_()
                projection.bias.fill_(100.)
        image = torch.rand(1, 4, 16, 16, requires_grad=True)
        reference = model(image).detach()
        with torch.autocast('cpu', dtype=torch.float16):
            prediction = model(image)
        self.assertTrue(torch.isfinite(prediction).all())
        torch.testing.assert_close(prediction.float(), reference, rtol=.02, atol=.01)
        prediction.float().square().mean().backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_mask_isolation_and_auxiliary_gradient_routing(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        model = SeeingDetector(channels=4, pretrained=False, dsconv_factory=FakeDSConv).train()
        x = torch.rand(2, 3, 64, 96)
        first = model(x, with_scm=True, mask=torch.zeros(2, 64, 96, dtype=torch.long))
        second = model(x, with_scm=True, mask=torch.ones(2, 64, 96, dtype=torch.long))
        for name in ('text', 'center', 'height', 'angle'):
            self.assertTrue(torch.equal(first[name], second[name]), name)
        self.assertEqual(first['text'].shape, (2, 2, 16, 24))
        self.assertEqual(first['sem_main'].shape[-2:], (4, 6))
        loss = (second['sem_main'] - second['scm_sem'].detach()).square().mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.semantic_adapter.parameters()))
        self.assertTrue(all(p.grad is None for p in model.scm.parameters()))
        model.zero_grad(set_to_none=True)
        result = model(x, with_scm=True, mask=torch.zeros(2, 64, 96, dtype=torch.long))
        result['scm_mask'].mean().backward()
        self.assertTrue(any(p.grad is not None for p in model.scm.encoder.parameters()))

    def test_inference_needs_no_labels_and_rejects_scm(self):
        torch.set_num_threads(1)
        model = SeeingDetector(channels=4, pretrained=False, dsconv_factory=FakeDSConv).eval()
        with torch.no_grad():
            prediction = model(torch.rand(1, 3, 64, 96))
        self.assertNotIn('scm_mask', prediction)
        with self.assertRaises(ValueError):
            model(torch.rand(1, 3, 64, 96), with_scm=True, mask=torch.zeros(1, 64, 96))


if __name__ == '__main__':
    unittest.main()
