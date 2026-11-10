"""DDP data/forward regression tests. No optimizer steps are performed."""
import unittest
import torch
from dexgrasp.train.distributed import GlobalBatchStream, cached_batch, FMLossModule
from dexgrasp.models.keypoint import KeypointModel
from dexgrasp.models.encoders import farthest_point_indices
from helpers import FixtureEncoder, small_config, condition


class CachedFixtureEncoder(FixtureEncoder):
    def forward(self, points, mask, *, indices=None):
        if indices is None:
            indices = farthest_point_indices(points, self.tokens)
        rows = torch.arange(len(points))[:, None]
        features = self.project(torch.cat((points, mask[..., None].float()), -1))
        return features[rows, indices], points[rows, indices]


class DistributedTests(unittest.TestCase):
    def test_full_2048_batch_is_partition_of_common_stream_and_resumes(self):
        serial = GlobalBatchStream(800, 2048, 1, 0, 42)
        ranks = [GlobalBatchStream(800, 256, 8, r, 42) for r in range(8)]
        for step in range(6):
            shards = [s.indices(step) for s in ranks]
            self.assertTrue(all(len(s) == 256 for s in shards))
            joined = sum(shards, [])
            self.assertEqual(joined, serial.indices(step))
            self.assertTrue(all(0 <= i < 800 for i in joined))
            self.assertEqual(GlobalBatchStream(800, 256, 8, 5, 42).indices(step), shards[5])
        first = serial.indices(0)
        self.assertEqual(sorted(first[:800]), list(range(800)))
        self.assertEqual(sorted(first[800:1600]), list(range(800)))
        self.assertGreater(len(first), len(set(first)))

    def test_cache_gather_keeps_condition_target_pairing_and_repeated_indices(self):
        cache = dict(condition={'x': torch.arange(30).reshape(10, 3)}, keypoints_gt=torch.arange(10))
        indices = [7, 2, 7, 0]
        batch = cached_batch(cache, indices)
        torch.testing.assert_close(batch['condition']['x'], cache['condition']['x'][indices])
        torch.testing.assert_close(batch['keypoints_gt'], torch.tensor(indices))

    def test_cached_fps_keeps_forward_and_gradients(self):
        torch.manual_seed(9)
        model = KeypointModel(small_config(), encoder_factory=CachedFixtureEncoder)
        c = condition(); state = torch.randn(2, 21, 3); t = torch.tensor([.1, .7])
        expected = model(state, t, model.encode(c)); expected.square().mean().backward()
        grads = {k: p.grad.clone() for k, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        cached = {**c, 'object_fps_indices': farthest_point_indices(c['object_points'], model.cfg.tokens),
                  'scene_fps_indices': farthest_point_indices(c['scene_points'], model.cfg.tokens)}
        actual = model(state, t, model.encode(cached)); actual.square().mean().backward()
        torch.testing.assert_close(actual, expected)
        for key, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, grads[key])

    def test_loss_wrapper_includes_encoder_gradients(self):
        model = KeypointModel(small_config(), encoder_factory=FixtureEncoder)
        loss = FMLossModule(model)(dict(condition=condition(), keypoints_gt=torch.randn(2, 21, 3)))
        loss.backward()
        self.assertTrue(model.object_encoder.project.weight.grad.abs().sum() > 0)
        self.assertTrue(model.scene_encoder.project.weight.grad.abs().sum() > 0)


if __name__ == '__main__':
    unittest.main()
