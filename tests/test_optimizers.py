"""Optimizer grouping/state tests; training optimizer steps remain disabled."""
from dataclasses import replace
import unittest
import torch
from dexgrasp.config import DEFAULT_RUN
from dexgrasp.models.keypoint import KeypointModel
from dexgrasp.train.optimizers import make_optimizer, parameter_groups
from dexgrasp.vendor.muon import _zeropower_via_newtonschulz, _adjust_lr
from helpers import small_config, FixtureEncoder


class OptimizerTests(unittest.TestCase):
    def test_partition_covers_each_trainable_parameter_once(self):
        for variant in ('full', 'object_only_h_less', 'object_direction_adaln_h_less'):
            model = KeypointModel(replace(small_config(), variant=variant), encoder_factory=FixtureEncoder)
            groups, names = parameter_groups(model)
            ids = [id(p) for g in groups.values() for p in g]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(set(ids), {id(p) for p in model.parameters()})
            self.assertIn('ids.weight', names['adamw'])
            self.assertIn('head.1.weight', names['adamw'])
            self.assertIn('embed.0.weight', names['adamw'])
            self.assertIn('blocks.0.ffn.0.weight', names['muon'])
            self.assertTrue(all(p.ndim == 2 for p in groups['muon']))

    def test_hybrid_state_roundtrip_and_wrong_group_rejected(self):
        model = KeypointModel(small_config(), encoder_factory=FixtureEncoder)
        cfg = {**DEFAULT_RUN, 'optimizer': 'muon'}
        optimizer = make_optimizer(model, cfg)
        state = optimizer.state_dict()
        make_optimizer(model, cfg).load_state_dict(state)
        state['names'] = {'muon': [], 'adamw': []}
        with self.assertRaises(ValueError): optimizer.load_state_dict(state)

    def test_newton_schulz_zero_finite_and_rectangular_shape(self):
        for shape in ((16, 8), (8, 16)):
            zero = _zeropower_via_newtonschulz(torch.zeros(shape), (3.4445, -4.775, 2.0315), 5, 1e-7)
            self.assertEqual(zero.shape, shape)
            self.assertTrue(torch.equal(zero, torch.zeros_like(zero)))
        self.assertEqual(_adjust_lr(.02, 'original', (16, 4)), .04)


if __name__ == '__main__':
    unittest.main()
