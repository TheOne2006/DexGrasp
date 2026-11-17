"""Global pairs must include cross-object, cross-scale and singleton-group records."""
from itertools import combinations
import math
import unittest

import numpy as np

from dexonomy.diversity import evaluate_pair_features


PARTS = {'joint': slice(None), 'translation': slice(3), 'rotation': slice(3, 12), 'hand': slice(12, None)}


def oracle(z, part=slice(None)):
    return math.fsum(math.exp(-float(np.sum((z[i, part] - z[j, part]) ** 2)))
                     for i, j in combinations(range(len(z)), 2)) / math.comb(len(z), 2)


class GlobalFeatureTests(unittest.TestCase):
    def test_direct_dense_oracle_all_global_blocks(self):
        z = np.random.default_rng(17).normal(scale=.1, size=(9, 75))
        result = evaluate_pair_features(z, batch_size=3)
        self.assertEqual((result['total_N'], result['total_U'], result['total_B']), (9, 36, 36))
        for b, sl in PARTS.items():
            mu = oracle(z, sl)
            self.assertAlmostEqual(result['scores'][b]['mu'], mu, places=14)
            self.assertAlmostEqual(result['scores'][b]['Q'], 1 - mu, places=14)
            self.assertEqual(result['scores'][b]['pair_mc_standard_error'], 0)

    def test_75d_global_pairs_include_all_three_landmark_coordinates(self):
        z = np.zeros((3,75)); z[1,74] = .3; z[2,12] = .5
        result = evaluate_pair_features(z)
        self.assertEqual(result['feature_dimension'],75)
        for block, part in PARTS.items():
            self.assertAlmostEqual(result['scores'][block]['mu'],oracle(z,part),places=14)

    def test_every_record_participates_even_if_one_condition_is_singleton(self):
        # Two coincident records plus one different record must include two
        # cross-condition pairs, not merely the identical within-condition pair.
        z = np.zeros((3, 75)); z[2, 0] = math.sqrt(math.log(2))
        actual = evaluate_pair_features(z)
        self.assertAlmostEqual(actual['dataset_diversity'], 1/3, places=14)
        self.assertEqual(actual['total_N'], 3)
        self.assertEqual(actual['total_U'], 3)

    def test_global_empty_and_singleton_only_are_na(self):
        for n in (0, 1):
            for mode in ('exact', 'sample'):
                result = evaluate_pair_features(np.zeros((n, 75)), mode=mode, num_pairs=10)
                self.assertEqual(result['status'], 'no_nonself_pairs')
                self.assertEqual(result['total_B'], 0)
                self.assertIsNone(result['dataset_diversity'])
                self.assertIsNone(result['dataset_collision_mean'])

    def test_identical_distant_and_overflow_limits(self):
        self.assertEqual(evaluate_pair_features(np.zeros((3, 75)))['dataset_diversity'], 0)
        z = np.zeros((2, 75)); z[0, 0] = 1e308; z[1, 0] = -1e308
        self.assertEqual(evaluate_pair_features(z)['dataset_diversity'], 1)

    def test_sample_is_one_global_stream_and_total_budget(self):
        z = np.random.default_rng(4).normal(scale=.2, size=(6, 75)); B = 777; seed = 83
        ids = np.random.default_rng(seed).integers(len(z) * (len(z) - 1), size=B)
        i, j = ids // (len(z) - 1), ids % (len(z) - 1); j += j >= i
        result = evaluate_pair_features(z, mode='sample', num_pairs=B, seed=seed, batch_size=13)
        self.assertEqual(result['total_B'], B)
        self.assertGreater(B, result['total_U'])
        for b, sl in PARTS.items():
            expected = np.exp(-np.sum((z[i, sl] - z[j, sl]) ** 2, axis=1)).mean()
            self.assertAlmostEqual(result['scores'][b]['mu'], expected, places=14)
        fallback = evaluate_pair_features(z, mode='sample', pairs_per_grasp=7)
        self.assertEqual(fallback['total_B'], 7 * len(z))

    def test_sample_reproducibility_batch_invariance_and_convergence(self):
        z = np.random.default_rng(5).normal(scale=.15, size=(15, 75))
        a = evaluate_pair_features(z, mode='sample', num_pairs=60001, seed=7, batch_size=17)
        b = evaluate_pair_features(z, mode='sample', num_pairs=60001, seed=7, batch_size=4096)
        exact = evaluate_pair_features(z)
        for block in PARTS:
            self.assertAlmostEqual(a['scores'][block]['mu'], b['scores'][block]['mu'], places=13)
            self.assertLess(abs(a['scores'][block]['mu'] - exact['scores'][block]['mu']), .003)

    def test_sample_standard_error_and_marginal_interval_match_independent_draws(self):
        z = np.random.default_rng(19).normal(scale=.1, size=(7, 75))
        count, seed = 113, 22
        # The known RNG selects ordered nonself pairs. Build the population
        # explicitly rather than reusing the evaluator's index-decoding helper.
        population = [(i, j) for i in range(len(z)) for j in range(len(z)) if i != j]
        selected = np.random.default_rng(seed).integers(len(population), size=count)
        result = evaluate_pair_features(z, mode='sample', num_pairs=count, seed=seed, batch_size=11)
        epsilon = math.sqrt(math.log(40)/(2*count))
        for block, sl in PARTS.items():
            values = np.array([math.exp(-float(np.sum((z[population[k][0], sl] -
                                                      z[population[k][1], sl])**2)))
                               for k in selected])
            score = result['scores'][block]
            self.assertAlmostEqual(score['pair_mc_standard_error'], values.std(ddof=1)/math.sqrt(count), places=14)
            np.testing.assert_allclose(score['mu_mc_95_interval'],
                [max(0., values.mean()-epsilon), min(1., values.mean()+epsilon)], rtol=0, atol=1e-14)
            np.testing.assert_allclose(score['Q_mc_95_interval'],
                [1-score['mu_mc_95_interval'][1], 1-score['mu_mc_95_interval'][0]], rtol=0, atol=1e-14)

    def test_one_draw_uncertainty_is_finite_and_conditional(self):
        z = np.zeros((3, 75)); z[2, 0] = 1
        r = evaluate_pair_features(z, mode='sample', num_pairs=1)
        self.assertIsNone(r['scores']['joint']['pair_mc_standard_error'])
        self.assertEqual(r['scores']['joint']['Q_mc_95_interval'], [0., 1.])
        self.assertIn('not subset', r['uncertainty']['scope'])

    def test_exact_order_invariance_and_srs_subset_unbiasedness(self):
        z = np.random.default_rng(6).normal(scale=.1, size=(5, 75))
        truth = evaluate_pair_features(z)['dataset_diversity']
        self.assertAlmostEqual(evaluate_pair_features(z[::-1])['dataset_diversity'], truth, places=14)
        for m in (2, 3, 4):
            estimates = [evaluate_pair_features(z[list(s)])['dataset_diversity'] for s in combinations(range(5), m)]
            self.assertAlmostEqual(float(np.mean(estimates)), truth, places=14)

    def test_uniform_replication_is_not_claimed_invariant(self):
        z = np.random.default_rng(8).normal(scale=.1, size=(4, 75)); copies = 3
        q = evaluate_pair_features(z)['dataset_diversity']
        copied = evaluate_pair_features(np.repeat(z, copies, axis=0))['dataset_diversity']
        self.assertAlmostEqual(copied, copies * 3 / 11 * q, places=14)

    def test_invalid_inputs_and_diagnostic_suppression(self):
        for z in [np.zeros(75), np.zeros((2, 14)), np.full((2, 75), np.nan)]:
            with self.assertRaises(ValueError): evaluate_pair_features(z)
        for kw in [{'mode': 'grouped'}, {'num_pairs': 0}, {'num_pairs': True}, {'seed': -1}, {'batch_size': 0}]:
            with self.assertRaises(ValueError): evaluate_pair_features(np.zeros((2, 75)), **kw)
        self.assertEqual(list(evaluate_pair_features(np.zeros((2, 75)), diagnostics=False)['scores']), ['joint'])



if __name__ == '__main__':
    unittest.main()
