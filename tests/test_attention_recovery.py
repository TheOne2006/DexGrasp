"""Numerical recovery regressions; forward/backward only, no optimizer updates."""
import copy
import unittest
import tempfile
from pathlib import Path
from dataclasses import asdict, replace
import torch
from torch.nn import functional as F
from dexgrasp.models.keypoint import centered_attention
from dexgrasp.train.recovery import validate_recovery


class AttentionRecoveryTests(unittest.TestCase):
    def test_checkpoint_load_preserves_numerical_backend_for_inference(self):
        from dexgrasp.models.keypoint import KeypointModel
        from dexgrasp.train.runner import load_model
        from helpers import FixtureEncoder, small_config
        model = KeypointModel(replace(small_config(), self_attention_backend='centered_fp32'),
                              encoder_factory=FixtureEncoder)
        previous = torch.backends.cuda.matmul.allow_tf32
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'model.pt'
                torch.save(dict(schema='dexgrasp-v1', method='fm', model_config=asdict(model.cfg),
                                model=model.state_dict()), path)
                torch.backends.cuda.matmul.allow_tf32 = True
                loaded, _ = load_model(path, 'cpu', encoder_factory=FixtureEncoder)
                self.assertTrue(all(b.self_attention.backend == 'centered_fp32' for b in loaded.blocks))
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                self.assertEqual(list(model.state_dict()), list(loaded.state_dict()))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous

    def test_centering_preserves_function_and_all_three_gradients(self):
        torch.manual_seed(12)
        xs = [torch.randn(2, 3, 21, 8, dtype=torch.float64, requires_grad=True) for _ in range(3)]
        upstream = torch.randn_like(xs[0])
        expected = F.scaled_dot_product_attention(*xs)
        actual = centered_attention(*xs)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        ga = torch.autograd.grad(actual, xs, upstream)
        ge = torch.autograd.grad(expected, xs, upstream)
        for a, e in zip(ga, ge):
            torch.testing.assert_close(a, e, rtol=1e-11, atol=1e-12)

    def test_equal_large_keys_have_uniform_output_and_zero_query_gradient(self):
        q = torch.full((1, 1, 21, 64), 16384., requires_grad=True)
        k = torch.full_like(q, 16384., requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        output = centered_attention(q, k, v)
        torch.testing.assert_close(output, v.mean(-2, keepdim=True).expand_as(v))
        dq, dk, dv = torch.autograd.grad(output, (q, k, v), torch.ones_like(output))
        self.assertEqual(float(dq.abs().max()), 0.)
        torch.testing.assert_close(dv, torch.ones_like(dv))
        self.assertTrue(torch.isfinite(dk).all())

    def test_recovery_rejects_recipe_or_identity_changes_and_nonfinite_state(self):
        cfg = dict(run_dir='/new/run', checkpoint_dir='/new/ckpt', model={'width': 32, 'self_attention_backend': 'centered_fp32'},
                   abort_grad_norm=100., batch_size=8, accumulation=1, lr=.001)
        old = {**cfg, 'run_dir': '/old/run', 'checkpoint_dir': '/old/ckpt', 'model': {'width': 32}}
        old.pop('abort_grad_norm')
        state = dict(schema='dexgrasp-v1', method='fm', config=old, data_identity={'id': 1}, runtime={'torch': 'same'},
                     distributed=dict(world_size=4, rank_rng=[{}]*4, global_batch_size=32),
                     hand_spec_sha256='hand', initial_checkpoint_sha256=None,
                     model={'w': torch.ones(1)}, optimizer={'state': {0: {'exp_avg': torch.zeros(1)}}})
        def validate(s=state, c=cfg):
            validate_recovery(s, c, {'id': 1}, {'torch': 'same'}, 4, 'hand', None)
        validate()
        for key, value in [('lr', .002), ('batch_size', 16), ('run_dir', '/old/run')]:
            with self.subTest(key=key), self.assertRaises(ValueError): validate(c={**cfg, key: value})
        bad = copy.deepcopy(state); bad['optimizer']['state'][0]['exp_avg'][0] = float('inf')
        with self.assertRaises(ValueError): validate(s=bad)


if __name__ == '__main__':
    unittest.main()
