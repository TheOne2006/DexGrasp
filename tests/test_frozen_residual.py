import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from dexgrasp.models.config import read_model_settings
from dexgrasp.models.frozen_residual import FrozenResidualConfig, FrozenResidualModel
from dexgrasp.models.registry import sample_model
from dexgrasp.train.distributed import FMLossModule
from helpers import FixtureEncoder, condition


class FrozenResidualTests(unittest.TestCase):
    def test_branch_sizes_and_unversioned_checkpoint_defaults(self):
        root = Path(__file__).resolve().parents[1] / 'configs' / 'frozen_residual'
        for name, depth, width, heads, count in (
            ('B_fixed_2x256', 2, 256, 4, 3_169_283),
            ('B_fixed_4x256', 4, 256, 4, 5_936_131),
            ('B_fixed', 4, 320, 5, 9_160_835),
        ):
            with self.subTest(name=name), torch.device('meta'):
                cfg = FrozenResidualConfig(**read_model_settings(root / f'{name}.json'))
                model = FrozenResidualModel(cfg, encoder_factory=FixtureEncoder)
                for branch in (model.pregrasp, model.squeeze):
                    self.assertEqual(len(branch.blocks), depth)
                    self.assertEqual(branch.state.out_features, width)
                    self.assertEqual(branch.blocks[0].self_attention.heads, heads)
                    self.assertEqual(sum(p.numel() for p in branch.parameters()), count)
                if name == 'B_fixed_2x256':
                    self.assertEqual(cfg, FrozenResidualConfig(grasp_model=cfg.grasp_model))

    def test_parent_endpoint_freeze_and_candidate_order(self):
        cfg = FrozenResidualConfig(grasp_model=dict(width=32, heads=4, tokens=8, depth=2,
                                   backbone_channels=16, variant='object_direction_adaln_h_less'), grasp_steps=2,
                                   residual_depth=4, residual_width=320, residual_heads=5)
        model = FrozenResidualModel(cfg, encoder_factory=FixtureEncoder).train()
        c = condition()
        poses = torch.randn(2, 3, 21, 3)
        with patch.object(model.grasp, 'features', wraps=model.grasp.features) as hidden:
            FMLossModule(model)(dict(condition=c, keypoints_gt=poses)).backward()
            anchor, t, context = hidden.call_args.args
            torch.testing.assert_close(anchor, poses[:, 1])
            torch.testing.assert_close(t, torch.ones(2))
        self.assertFalse(model.grasp.training)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.grasp.parameters()))
        for branch in (model.pregrasp, model.squeeze):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters()))
        self.assertTrue(set(model.pregrasp.parameters()).isdisjoint(model.squeeze.parameters()))
        model.eval()
        noise = torch.randn(4, 3, 21, 3)
        calls = model.grasp.object_encoder.calls
        with patch.object(model.grasp, 'features', wraps=model.grasp.features) as hidden:
            poses = sample_model(model, c, steps=2, candidates=2, noise=noise)
            self.assertEqual(hidden.call_count, 3)
            final_grasp, t, _ = hidden.call_args.args
            torch.testing.assert_close(poses[:, 1], final_grasp)
            torch.testing.assert_close(t, torch.ones(4))
        self.assertEqual(model.grasp.object_encoder.calls, calls+1)
        uncached = sample_model(model, c, steps=2, candidates=2, noise=noise, cache_kv=False)
        torch.testing.assert_close(uncached, poses, rtol=1e-5, atol=2e-6)
        changed = noise.clone()
        changed[:, 0] += .2
        other = sample_model(model, c, steps=2, candidates=2, noise=changed, cache_kv=False)
        torch.testing.assert_close(other[:, 1:], poses[:, 1:], rtol=1e-5, atol=2e-6)
        self.assertFalse(torch.allclose(other[:, 0], poses[:, 0]))

    @unittest.skipUnless(os.environ.get('DEXGRASP_GPU_TESTS') == '1' and torch.cuda.is_available(),
                         'explicit GPU regression requested')
    def test_independent_cuda_streams_match_serial_integration(self):
        cfg = FrozenResidualConfig(grasp_model=dict(width=32, heads=4, tokens=8, depth=2,
                                   backbone_channels=16, variant='object_direction_adaln_h_less'), grasp_steps=2)
        model = FrozenResidualModel(cfg, encoder_factory=FixtureEncoder).cuda().eval()
        c = condition(device='cuda')
        noise = torch.randn(4, 3, 21, 3, device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            serial = sample_model(model, c, steps=3, candidates=2, noise=noise, parallel_branches=False)
            concurrent = sample_model(model, c, steps=3, candidates=2, noise=noise, parallel_branches=True)
        torch.testing.assert_close(concurrent, serial)
