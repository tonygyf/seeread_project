"""CPU contracts; set SELF_DACE_SOURCE_ROOT to also test official public weights."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from seeread.v5.enhancers import build_enhancer


class EnhancerContracts(unittest.TestCase):
    def test_identity_needs_no_external_source_and_preserves_gradient(self):
        enhancer = build_enhancer(SimpleNamespace(type="identity", source_root="/missing"))
        x = torch.rand(2, 3, 9, 15, requires_grad=True)
        self.assertIs(enhancer(x), x)
        enhancer(x).sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))

    def test_invalid_inputs_and_unsupported_methods_fail_explicitly(self):
        enhancer = build_enhancer({"type": "identity"})
        for x in (torch.rand(1, 1, 5, 5), torch.full((1, 3, 5, 5), float("nan")),
                  torch.full((1, 3, 5, 5), 1.01)):
            with self.assertRaises(ValueError):
                enhancer(x)
        with self.assertRaises(TypeError):
            enhancer(torch.zeros(1, 3, 5, 5, dtype=torch.uint8))
        for kind in ("rllie", "iae"):
            with self.assertRaises(NotImplementedError):
                build_enhancer({"type": kind})
        with self.assertRaises(ValueError):
            build_enhancer({"type": "srdace"})
        with self.assertRaises(ValueError):
            build_enhancer({"type": "unknown"})


@unittest.skipUnless(os.environ.get("SELF_DACE_SOURCE_ROOT"), "official checkout not configured")
class OfficialSelfDACEContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.root = Path(os.environ["SELF_DACE_SOURCE_ROOT"])

    def _reference(self, stage):
        prefix = self.root / "codes_SelfDACE/old_version" / f"stage{stage}"
        spec = importlib.util.spec_from_file_location(f"official_reference_stage{stage}", prefix / "model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        light = module.light_net().eval()
        light.load_state_dict(torch.load(prefix / "snapshots_light/pre_train_1.pth", map_location="cpu", weights_only=True))
        denoise = None
        if stage == 2:
            denoise = module.denoise_net().eval()
            denoise.load_state_dict(torch.load(prefix / "snapshots_denoise/pre_train_2.pth", map_location="cpu", weights_only=True)["net"])
        return light, denoise

    def test_both_stages_equal_upstream_and_frozen_input_gradient_survives(self):
        for stage in (1, 2):
            with self.subTest(stage=stage):
                enhancer = build_enhancer(dict(type="srdace", source_root=str(self.root),
                                              stage=stage, clamp_output=False))
                reference, denoise = self._reference(stage)
                x = torch.linspace(0.01, 0.3, 2 * 3 * 9 * 15).reshape(2, 3, 9, 15).requires_grad_()
                expected = reference(x)[0]
                if denoise is not None:
                    expected = denoise(expected)
                actual = enhancer(x)
                self.assertEqual(actual.shape, x.shape)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertGreater(x.grad.abs().sum().item(), 0)
                self.assertTrue(all(p.grad is None and not p.requires_grad for p in enhancer.parameters()))
                enhancer.train()
                self.assertTrue(all(not m.training for m in enhancer.modules()))
                self.assertEqual(len(enhancer.provenance["checkpoint_sha256"]), 64)
                clamped = build_enhancer(dict(type="srdace", source_root=str(self.root), stage=stage))
                torch.testing.assert_close(clamped(x), expected.clamp(0, 1), rtol=0, atol=0)

    def test_trainable_official_light_parameters_receive_gradients(self):
        enhancer = build_enhancer(dict(type="self_dace", source_root=str(self.root), stage=1, freeze=False))
        x = torch.rand(1, 3, 7, 11) * 0.3
        enhancer(x).mean().backward()
        gradients = [p.grad for p in enhancer.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)

    def test_incompatible_checkpoint_is_rejected(self):
        with self.assertRaises(RuntimeError):
            build_enhancer(dict(type="srdace", source_root=str(self.root), stage=1,
                                checkpoint="codes_SelfDACE/old_version/stage2/snapshots_denoise/pre_train_2.pth"))


if __name__ == "__main__":
    unittest.main()
