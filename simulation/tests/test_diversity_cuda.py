"""Backend and prefetch parity without implicitly acquiring a GPU in CPU CI."""
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dexonomy.diversity import evaluate_pair_features
from dexonomy.diversity._cuda import accumulate, _prefetched
from dexonomy.diversity.metrics import _sample_pairs, _exact_pairs


class PrefetchTests(unittest.TestCase):
    def test_original_pcg64_stream_is_preserved(self):
        n, count, seed = 53, 7891, 149
        ids = np.random.default_rng(seed).integers(n * (n - 1), size=count)
        first, second = ids // (n - 1), ids % (n - 1)
        second += second >= first
        for batch_size in (1, 377, 10000):
            batches = list(_prefetched(_sample_pairs(n, count, batch_size, seed)))
            np.testing.assert_array_equal(np.concatenate([b[0] for b in batches]), first)
            np.testing.assert_array_equal(np.concatenate([b[1] for b in batches]), second)

    def test_producer_error_reaches_consumer_and_early_close_is_safe(self):
        def broken():
            yield (1, 2)
            raise ValueError('producer failure')
        stream = _prefetched(broken())
        self.assertEqual(next(stream), (1, 2))
        with self.assertRaisesRegex(ValueError, 'producer failure'):
            next(stream)
        stream = _prefetched(iter(range(100)))
        self.assertEqual(next(stream), 0)
        stream.close()


class BackendTests(unittest.TestCase):
    def test_torch_reduction_matches_numpy_for_shapes_modes_and_partial_batches(self):
        for dim in (15, 75, 78):
            z = np.random.default_rng(dim).normal(0, .1, (23, dim))[::-1]
            for mode in ('sample', 'exact'):
                reference = evaluate_pair_features(z, mode=mode, num_pairs=1003, seed=17, batch_size=83)
                for prefetch in (False, True):
                    batches = _sample_pairs(len(z), 1003, 127, 17) if mode == 'sample' else _exact_pairs(len(z), 127)
                    count, mean, m2, execution = accumulate(z, batches, device='cpu', prefetch=prefetch)
                    self.assertEqual(count, reference['total_B'])
                    for j, block in enumerate(reference['scores']):
                        self.assertAlmostEqual(1 - mean[j], reference['scores'][block]['Q'], places=13)
                        if mode == 'sample':
                            self.assertAlmostEqual(np.sqrt(m2[j] / (count-1) / count), reference['scores'][block]['pair_mc_standard_error'], places=13)
                    self.assertEqual(execution['dtype'], 'float64')

    def test_public_dispatch_and_default_batches(self):
        z = np.random.default_rng(71).normal(0, .1, (17, 75))
        cpu = evaluate_pair_features(z, mode='sample', num_pairs=1003, seed=19)
        self.assertEqual(cpu['batch_size'], 16384)
        # Exercise the public CUDA branch using Torch CPU in hardware-free CI.
        def on_cpu(features, batches, **kwargs):
            self.assertEqual(kwargs['device'], 'cuda:0')
            return accumulate(features, batches, device='cpu', prefetch=kwargs['prefetch'])
        with patch('dexonomy.diversity._cuda.accumulate', side_effect=on_cpu):
            gpu = evaluate_pair_features(z, mode='sample', num_pairs=1003, seed=19, device='cuda:0')
        self.assertEqual(gpu['batch_size'], 262144)
        self.assertTrue(gpu['execution']['pair_prefetch'])
        for block in cpu['scores']:
            self.assertAlmostEqual(gpu['scores'][block]['Q'], cpu['scores'][block]['Q'], places=13)

    def test_invalid_execution_options_and_empty_cuda_pool(self):
        for kwargs in ({'device': 'auto'}, {'device': None}, {'device': 'cuda:-1'}, {'prefetch': 1}):
            with self.assertRaises(ValueError):
                evaluate_pair_features(np.zeros((2, 75)), **kwargs)
        with patch('dexonomy.diversity._cuda.accumulate', side_effect=AssertionError('must not acquire device')):
            result = evaluate_pair_features(np.zeros((1, 75)), device='cuda')
        self.assertEqual(result['execution']['backend'], 'none')
        self.assertIsNone(result['dataset_diversity'])

    def test_explicit_cuda_unavailable_does_not_fall_back(self):
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'unavailable'):
                evaluate_pair_features(np.zeros((2, 75)), device='cuda')


class CachedCliTests(unittest.TestCase):
    def test_feature_cli_scores_and_provenance(self):
        from dexonomy.diversity.__main__ import main
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / 'weighted.npy', Path(tmp) / 'result.json'
            z = np.random.default_rng(99).normal(0, .1, (31, 75)); np.save(source, z)
            main(['--features', str(source), '--mode', 'sample', '--num-pairs', '1003',
                  '--seed', '17', '--output', str(output)])
            result = json.loads(output.read_text())
            reference = evaluate_pair_features(z, mode='sample', num_pairs=1003, seed=17)
            self.assertEqual(result['scores'], reference['scores'])
            self.assertTrue(result['feature_source']['weights_already_applied'])
            self.assertGreaterEqual(result['elapsed_seconds'], result['execution']['pair_seconds'])
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['--features', str(source), '--output', str(output)])

    def test_cli_requires_one_input_kind_and_native_asset_root(self):
        from dexonomy.diversity.__main__ import main
        for argv in ([], ['native'], ['native', '--features', 'features.npy'],
                     ['--features', 'features.npy', '--save-features', 'output.npy']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(argv)


if __name__ == '__main__':
    unittest.main()
