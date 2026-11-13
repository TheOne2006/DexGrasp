"""Optional GPU kernel equivalence checks; no optimizer updates."""
import copy
import os
import unittest
from dataclasses import replace
import torch
from dexgrasp.config import DEFAULT_RUN
from dexgrasp.models.encoders import farthest_point_indices
from dexgrasp.models.keypoint import ConditionalBlock, KeypointModel
from dexgrasp.train.execution import configure_execution
from helpers import small_config, FixtureEncoder


@unittest.skipUnless(os.environ.get('DEXGRASP_GPU_TESTS') == '1' and torch.cuda.is_available(),
                     'explicit DEXGRASP_GPU_TESTS=1 required')
class ExecutionTests(unittest.TestCase):
    def test_selected_sparse_gather_matches_slice_forward_and_backward(self):
        import MinkowskiEngine as ME
        from dexgrasp.models.encoders import selected_point_features
        torch.manual_seed(42)
        points = torch.randn(2, 1024, 3, device='cuda')*4
        coordinates = torch.cat((torch.arange(2, device='cuda').repeat_interleave(1024)[:, None],
                                 points.reshape(-1, 3)), -1)
        field = ME.TensorField(features=torch.randn(2048, 7, device='cuda'), coordinates=coordinates,
                               quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
        sparse = field.sparse()
        features = sparse.F.detach().requires_grad_(True)
        output = ME.SparseTensor(features=features, coordinate_map_key=sparse.coordinate_map_key,
                                 coordinate_manager=sparse.coordinate_manager)
        indices = farthest_point_indices(points, 128)
        rows = torch.arange(2, device='cuda')[:, None]
        expected = output.slice(field).F.reshape(2, 1024, -1)[rows, indices]
        actual = selected_point_features(output, field, indices, 1024)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        gradient = torch.randn_like(expected)
        grad_expected, = torch.autograd.grad(expected, features, gradient)
        grad_actual, = torch.autograd.grad(actual, features, gradient)
        torch.testing.assert_close(grad_actual, grad_expected, rtol=1e-5, atol=1e-6)

    def test_fused_fps_matches_reference_including_ties(self):
        from dexgrasp.models.fps_triton import farthest_point_indices_triton
        torch.manual_seed(42)
        for size in (17, 1024, 2048):
            for kind in ('random', 'zero', 'duplicates'):
                points = torch.randn(8, size, 3, device='cuda')
                if kind != 'random':
                    points.zero_()
                if kind == 'duplicates':
                    points[:, ::2] = 1
                count = min(128, size)
                expected = farthest_point_indices(points, count)
                actual = farthest_point_indices_triton(points, count)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_compiled_block_preserves_outputs_gradients_and_state_keys(self):
        for variant in ('object_only_h_less', 'object_direction_adaln_h_less'):
            cfg = replace(small_config(), variant=variant)
            torch.manual_seed(42)
            eager = ConditionalBlock(cfg).cuda()
            compiled = copy.deepcopy(eager)
            compiled.checkpoint_direction_bias = True
            compiled.compile(fullgraph=True, dynamic=False)
            x = torch.randn(2, 21, cfg.width, device='cuda')
            condition = torch.randn(2, cfg.width, device='cuda')
            tokens = (torch.randn(2, cfg.tokens, cfg.width, device='cuda'),)
            geometry = () if cfg.direction_adaln else (torch.randn(2, 21, cfg.tokens, 2, device='cuda'),)
            expected = eager(x, condition, tokens, geometry)
            expected.square().mean().backward()
            actual = compiled(x, condition, tokens, geometry)
            actual.square().mean().backward()
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
            self.assertEqual(list(compiled.state_dict()), list(eager.state_dict()))
            for (name, p), (_, q) in zip(eager.named_parameters(), compiled.named_parameters()):
                self.assertIsNotNone(p.grad, name)
                self.assertIsNotNone(q.grad, name)
                torch.testing.assert_close(q.grad, p.grad, rtol=3e-4, atol=2e-5)

    def test_bfloat16_compilation_and_recompute_preserve_numerics(self):
        for variant in ('object_only_h_less', 'object_direction_adaln_h_less'):
            with self.subTest(variant=variant):
                cfg = replace(small_config(), variant=variant)
                torch.manual_seed(71)
                eager = ConditionalBlock(cfg).cuda()
                compiled = copy.deepcopy(eager)
                compiled.checkpoint_direction_bias = True
                compiled.compile(fullgraph=True, dynamic=False)
                x = torch.randn(2, 21, cfg.width, device='cuda')
                condition = torch.randn(2, cfg.width, device='cuda')
                tokens = (torch.randn(2, cfg.tokens, cfg.width, device='cuda'),)
                geometry = () if cfg.direction_adaln else (torch.randn(2, 21, cfg.tokens, 2, device='cuda'),)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    expected = eager(x, condition, tokens, geometry)
                    actual = compiled(x, condition, tokens, geometry)
                expected.float().square().mean().backward()
                actual.float().square().mean().backward()
                # BF16 fusion changes rounding; bound normwise error rather than
                # relative error at individual near-zero entries.
                pairs = [('output', actual, expected)] + [
                    (name, q.grad, p.grad) for (name, p), (_, q) in
                    zip(eager.named_parameters(), compiled.named_parameters())]
                for name, actual_value, expected_value in pairs:
                    self.assertIsNotNone(actual_value, name)
                    self.assertIsNotNone(expected_value, name)
                    if name.endswith('biases.0.2.bias'):
                        # A constant offset per attention head cancels in softmax.
                        # Its exact gradient is zero; BF16 leaves roundoff residuals.
                        self.assertLessEqual(float(actual_value.float().norm()), 1e-5, name)
                        self.assertLessEqual(float(expected_value.float().norm()), 1e-5, name)
                        continue
                    error = (actual_value.float()-expected_value.float()).norm()
                    bound = .02*expected_value.float().norm()+1e-6
                    self.assertLessEqual(float(error), float(bound), name)


class ExecutionConfigTests(unittest.TestCase):
    def test_configuration_does_not_rename_checkpoint_parameters(self):
        model = KeypointModel(small_config(), encoder_factory=FixtureEncoder)
        before = list(model.state_dict())
        configure_execution(model, {**DEFAULT_RUN, 'compile_transformer': True})
        self.assertEqual(list(model.state_dict()), before)


if __name__ == '__main__':
    unittest.main()
