"""CPU upstream parity; integration tests need DSCNET_SOURCE_ROOT and einops."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from seeread.v5.dsconv import OfficialDSConv, DSCONV_SOURCE_PATH


class DSConvConfiguration(unittest.TestCase):
    def test_missing_source_fails_without_importing_optional_operator(self):
        with self.assertRaisesRegex(ValueError, "source_root"):
            OfficialDSConv(3, 8)
        with self.assertRaises(FileNotFoundError):
            OfficialDSConv(3, 8, source_root="/no/dscnet/checkout")

    def test_invalid_geometry_is_rejected_before_external_import(self):
        for options in (dict(kernel_size=4), dict(morph=2), dict(extend_scope=-1)):
            with self.assertRaises(ValueError):
                OfficialDSConv(3, 8, source_root="/unused", **options)


@unittest.skipUnless(os.environ.get("DSCNET_SOURCE_ROOT"), "official DSCNet checkout not configured")
class OfficialDSConvParity(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.root = Path(os.environ["DSCNET_SOURCE_ROOT"])
        spec = importlib.util.spec_from_file_location("dsconv_direct_reference", self.root / DSCONV_SOURCE_PATH)
        self.reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.reference)

    def _pair(self, morph, if_offset=True):
        torch.manual_seed(41 + morph)
        adapted = OfficialDSConv(3, 8, kernel_size=5, morph=morph,
                                 if_offset=if_offset, source_root=str(self.root))
        direct = self.reference.DSConv_pro(3, 8, kernel_size=5, morph=morph, device="cpu")
        direct.load_state_dict(adapted.operator.state_dict(), strict=True)
        return adapted, direct

    def test_large_deformed_convolution_stays_finite_under_amp(self):
        for morph in (0, 1):
            adapted, _ = self._pair(morph)
            conv = adapted.operator.dsc_conv_x if morph == 0 else adapted.operator.dsc_conv_y
            with torch.no_grad():
                conv.weight.fill_(50000.)
            image = torch.rand(2, 3, 8, 11, requires_grad=True)
            reference = adapted(image).detach()
            with torch.autocast('cpu', dtype=torch.float16):
                prediction = adapted(image)
            self.assertTrue(torch.isfinite(prediction).all())
            torch.testing.assert_close(prediction.float(), reference, rtol=.05, atol=.05)
            prediction.float().square().mean().backward()
            self.assertTrue(torch.isfinite(conv.weight.grad).all())
            self.assertTrue(torch.isfinite(image.grad).all())

    def test_forward_input_and_offset_gradients_match_both_axes(self):
        for morph in (0, 1):
            with self.subTest(morph=morph):
                adapted, direct = self._pair(morph)
                x = torch.randn(2, 3, 8, 11, requires_grad=True)
                ref_x = x.detach().clone().requires_grad_()
                # A stale constructor device must not survive Module.to()/input movement.
                adapted.operator.device = torch.device("cuda:7")
                actual, expected = adapted(x), direct(ref_x)
                self.assertEqual(actual.shape, (2, 8, 8, 11))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                weights = torch.linspace(0.1, 1, actual.numel()).reshape_as(actual)
                (actual * weights).mean().backward()
                (expected * weights).mean().backward()
                torch.testing.assert_close(x.grad, ref_x.grad, rtol=0, atol=0)
                for name in ("weight", "bias"):
                    grad = getattr(adapted.operator.offset_conv, name).grad
                    reference_grad = getattr(direct.offset_conv, name).grad
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(grad.abs().sum().item(), 0)
                    torch.testing.assert_close(grad, reference_grad, rtol=0, atol=0)

    def test_batch_samples_remain_independent(self):
        for morph in (0, 1):
            adapted, _ = self._pair(morph)
            adapted.eval()
            first = torch.randn(1, 3, 8, 11)
            second = torch.randn(1, 3, 8, 11) * 3 + 5
            batch = adapted(torch.cat([first, second]))
            torch.testing.assert_close(batch[0], adapted(first)[0], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(batch[1], adapted(second)[0], rtol=1e-5, atol=1e-6)
            self.assertFalse(torch.allclose(batch[0], batch[1]))

    def test_disabled_offsets_equal_upstream_zero_offsets_and_ignore_predictor(self):
        for morph in (0, 1):
            adapted, direct = self._pair(morph, if_offset=False)
            x = torch.randn(2, 3, 8, 11, requires_grad=True)
            # Zero AFTER GN because its learned affine shift need not be zero.
            hook = direct.gn_offset.register_forward_hook(lambda _m, _i, out: torch.zeros_like(out))
            try:
                expected = direct(x)
            finally:
                hook.remove()
            actual = adapted(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            with torch.no_grad():
                adapted.operator.offset_conv.weight.fill_(100)
                adapted.operator.gn_offset.bias.fill_(10)
            torch.testing.assert_close(adapted(x), actual, rtol=0, atol=0)
            actual.square().mean().backward()
            self.assertIsNone(adapted.operator.offset_conv.weight.grad)
            self.assertGreater(x.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
