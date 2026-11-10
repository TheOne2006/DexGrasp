import importlib.util
import unittest
from unittest.mock import patch
import torch
from dexgrasp.models.keypoint import KeypointModel, Attention, direction_geometry
from dexgrasp.models.encoders import farthest_point_indices, GeometryEncoder
from dexgrasp.train.objectives import (fm_pair, fm_loss, sample_keypoints, ode_trajectory,
                                       interval_pairs, flowmap_loss)
from dexgrasp.test.inference import predict
from helpers import FixtureEncoder, condition, small_config


class ModelTests(unittest.TestCase):
    def model(self, method='fm'):
        return KeypointModel(small_config(), method, encoder_factory=FixtureEncoder)

    def test_fm_endpoints_and_gradients_all_branches(self):
        clean, noise = torch.randn(2, 21, 3), torch.randn(2, 21, 3)
        state, t, velocity = fm_pair(clean, noise=noise, time=torch.tensor([0., 1.]))
        torch.testing.assert_close(state[0], noise[0])
        torch.testing.assert_close(state[1], clean[1])
        torch.testing.assert_close(velocity, clean-noise)
        model = self.model()
        loss = fm_loss(model, dict(condition=condition(), keypoints_gt=clean))
        loss.backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
        self.assertIsNot(model.object_encoder, model.scene_encoder)
        self.assertIsNot(model.blocks[0].biases[0], model.blocks[1].biases[0])

    def test_direction_geometry_matches_axial_distance_and_rotation(self):
        state, centers, d = torch.randn(2, 21, 3), torch.randn(2, 8, 3), torch.randn(2, 3)
        d = d/d.norm(dim=-1, keepdim=True)
        geom = direction_geometry(state, centers, d)
        r = centers[:, None]-state[:, :, None]
        torch.testing.assert_close(geom[..., 0], (r*d[:, None, None]).sum(-1))
        torch.testing.assert_close(geom[..., 1], torch.cross(r, d[:, None, None].expand_as(r), dim=-1).norm(dim=-1))
        rot, _ = torch.linalg.qr(torch.randn(3, 3))
        torch.testing.assert_close(geom, direction_geometry(state@rot, centers@rot, d@rot), atol=1e-6, rtol=1e-5)

    def test_attention_matches_explicit_softmax(self):
        attn = Attention(32, 4)
        x, g = torch.randn(2, 21, 32), torch.randn(2, 8, 32)
        bias = torch.randn(2, 4, 21, 8, requires_grad=True)
        q, (k, v) = attn.split(attn.query(x)), attn.project_kv(g)
        expected = attn.output(((q@k.transpose(-1, -2)/8**.5+bias).softmax(-1)@v).transpose(1, 2).flatten(2))
        actual = attn(x, g, bias)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
        actual.square().mean().backward()
        self.assertGreater(float(bias.grad.abs().sum()), 0.)

    def test_kv_cache_exact_and_encoding_once_for_candidates(self):
        model, c, noise = self.model(), condition(), torch.randn(6, 21, 3)
        cached = sample_keypoints(model, c, steps=3, candidates=3, noise=noise, cache_kv=True)
        self.assertEqual(model.object_encoder.calls, 1)
        self.assertEqual(model.scene_encoder.calls, 1)
        uncached = sample_keypoints(model, c, steps=3, candidates=3, noise=noise, cache_kv=False)
        torch.testing.assert_close(cached, uncached, atol=2e-6, rtol=1e-5)
        self.assertTrue(model.training)
        with self.assertRaises(ValueError):
            model.encode(c, cache_kv=True)

    def test_flowmap_identity_and_train_gradients(self):
        model, teacher, c = self.model('flowmap'), self.model('fm').requires_grad_(False), condition()
        x, t = torch.randn(2, 21, 3), torch.tensor([.1, .8])
        torch.testing.assert_close(model(x, t, model.encode(c), t), x, atol=0, rtol=0)
        loss = flowmap_loss(model, teacher, dict(condition=c, keypoints_gt=x), steps=2)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_teacher_ode_not_linear_interpolation_and_pair_identity(self):
        class Exponential(torch.nn.Module):
            method = 'fm'
            def encode(self, c, cache_kv=True): return None
            def forward(self, x, t, c): return x
        teacher = Exponential()
        path = ode_trajectory(teacher, {}, torch.ones(4, 21, 3), steps=4, solver='heun')
        expected = (1+.25+.25**2/2)**4
        torch.testing.assert_close(path[:, -1], torch.full((4, 21, 3), expected))
        self.assertFalse(torch.allclose(path[:, 2], (path[:, 0]+path[:, -1])/2))
        a, b, ta, tb = interval_pairs(path, 1.)
        torch.testing.assert_close(a, path[:, 0])
        torch.testing.assert_close(b, path[:, -1])
        self.assertTrue((ta == 0).all() and (tb == 1).all())

    def test_interval_pairs_never_mix_batch_or_trajectory(self):
        path = torch.arange(10)[None, :, None, None].expand(40, -1, 21, 3).float()
        path = path + torch.arange(40)[:, None, None, None]*100
        a, b, ta, tb = interval_pairs(path, .25)
        self.assertTrue((ta < tb).all())
        torch.testing.assert_close(a[:, 0, 0], ta*9+torch.arange(40)*100)
        torch.testing.assert_close(b[:, 0, 0], tb*9+torch.arange(40)*100)

    def test_flowmap_multi_step_is_mapping_not_velocity_update(self):
        model = self.model('flowmap')
        def mapping(state, a, ctx, b):
            return state + (b-a)[:, None, None]*2
        with patch.object(model, 'forward', side_effect=mapping) as call:
            result = sample_keypoints(model, condition(), steps=4, noise=torch.zeros(2, 21, 3))
        torch.testing.assert_close(result, torch.full_like(result, 2))
        self.assertEqual(call.call_count, 4)

    def test_condition_checks_and_direction_normalization(self):
        model, c = self.model(), condition()
        c['grasp_direction_cmd'] *= 7
        ctx = model.encode(c)
        torch.testing.assert_close(ctx.direction.norm(dim=-1), torch.ones(2))
        for key, replacement in [('palm_gap', torch.full((2, 1), -1.)),
                                  ('grasp_direction_cmd', torch.zeros(2, 3)),
                                  ('scene_mask', torch.zeros(2, 2048))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                model.encode({**c, key: replacement})

    def test_fps_duplicates_do_not_repeat_indices(self):
        indices = farthest_point_indices(torch.zeros(2, 9, 3), 8)
        self.assertEqual(len(indices[0].unique()), 8)

    def test_new_scene_inference_normalization_and_translation(self):
        model, raw, noise = self.model('flowmap'), condition(), torch.randn(4, 21, 3)
        result = predict(model, raw, candidates=2, noise=noise)
        self.assertEqual(result.shape, (2, 2, 21, 3))
        shift = torch.tensor([.1, -.2, .3])
        moved = {**raw, 'object_points': raw['object_points']+shift, 'scene_points': raw['scene_points']+shift}
        torch.testing.assert_close(predict(model, moved, candidates=2, noise=noise), result+shift, atol=2e-6, rtol=1e-5)
        raw['scene_mask'] *= 2
        with self.assertRaises(ValueError): predict(model, raw)

    def test_invalid_time_intervals_rejected(self):
        model = self.model('flowmap')
        context = model.encode(condition())
        with self.assertRaises(ValueError):
            model(torch.randn(2, 21, 3), torch.full((2,), .8), context, torch.full((2,), .2))

    def test_bfloat16_mapping_backward_has_finite_gradients(self):
        model, c = self.model('flowmap'), condition()
        state, a, b = torch.randn(2, 21, 3), torch.zeros(2), torch.ones(2)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            output = model(state, a, model.encode(c), b)
            loss = output.square().mean()
        loss.backward()
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    @unittest.skipUnless(importlib.util.find_spec('MinkowskiEngine'), 'MinkowskiEngine unavailable: real sparse backend not exercised')
    def test_real_minkunet_forward_backward(self):
        encoder = GeometryEncoder(small_config()).eval()
        points = torch.randn(2, 1024, 3)*.1
        tokens, centers = encoder(points, torch.ones(2, 1024))
        self.assertEqual(tokens.shape, (2, 8, 32))
        self.assertEqual(centers.shape, (2, 8, 3))
        tokens.square().mean().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in encoder.parameters()))


if __name__ == '__main__':
    unittest.main()
