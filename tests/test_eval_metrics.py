import tempfile
import unittest
from pathlib import Path
import numpy as np

from helpers import hand_fixture
from dexgrasp.geometry.hand import URDFHand
from dexgrasp.test.metrics import reconstruction_metrics, joint_diversity, decode_metrics, summarize
from dexgrasp.test.formal import select_conditions, official_hand, compare, write_json
from dexgrasp.data.dataset import sha256


class EvaluationMetricsTests(unittest.TestCase):
    def test_mean_max_and_rmse_are_distinct_and_in_mm(self):
        k = np.zeros((21,3)); rebuilt = k.copy(); rebuilt[0,0] = .021
        result = reconstruction_metrics(k, rebuilt)
        self.assertAlmostEqual(result['mean_keypoint_error_mm'], 1.)
        self.assertAlmostEqual(result['max_keypoint_error_mm'], 21.)
        self.assertAlmostEqual(result['reconstruction_rmse_mm'], np.sqrt(21))
        self.assertFalse(result['fit_success'])
        rebuilt = np.full((21,3), 0.); rebuilt[:,0] = .005
        self.assertTrue(reconstruction_metrics(k, rebuilt)['fit_success'])

    def test_diversity_requires_two_and_uses_sample_std(self):
        self.assertIsNone(joint_diversity([]))
        self.assertIsNone(joint_diversity(np.zeros((1,22))))
        self.assertEqual(joint_diversity(np.zeros((8,22))), 0.)
        self.assertAlmostEqual(joint_diversity([np.zeros(22), np.ones(22)]), np.sqrt(.5))

    def test_failures_stay_in_rates_and_undefined_errors_remain_null(self):
        rows = [dict(object_id='a', condition_id='x', decoded=True, finite_output=True,
                     fit_success=True, direction_pass=True, mean_keypoint_error_mm=2.,
                     max_keypoint_error_mm=3., direction_error_deg=4., q=[0.]*22),
                dict(object_id='a', condition_id='x', decoded=False, finite_output=False,
                     fit_success=False, direction_pass=False),
                dict(object_id='b', condition_id='y', decoded=False, finite_output=False,
                     fit_success=False, direction_pass=False)]
        out = summarize(rows, bootstrap_repeats=50)
        self.assertEqual(out['macro']['ik_fit_success_rate']['mean'], .25)
        self.assertEqual(out['macro']['mean_keypoint_error_mm']['mean'], 2.)
        self.assertEqual(out['macro']['mean_keypoint_error_mm_coverage']['mean'], .25)
        self.assertIsNone(out['macro']['joint_angle_diversity_rad']['mean'])
        self.assertIsNone(out['physical_success_rate'])
        self.assertEqual(out['candidates'], 3)

    def test_selection_is_order_independent_and_deduplicates(self):
        obs = np.array([0,0,0,1,1,1]); d = np.tile([0.,1.,0.],(6,1))
        candidates = [(i,'a' if i<3 else 'b') for i in range(6)]
        a = select_conditions(candidates, obs, d, ['a','b'], count=4, seed=7)
        b = select_conditions(reversed(candidates), obs, d, ['a','b'], count=4, seed=7)
        self.assertEqual(a,b); self.assertEqual(len(a),2)
        with self.assertRaises(ValueError):
            select_conditions([], obs, d, ['a'], count=4, seed=7)

    def test_all_failures_and_singleton_diversity_are_json_safe(self):
        import json
        rows = [dict(object_id='a', condition_id='x', decoded=False, fit_success=False,
                     direction_pass=False, finite_output=False)]
        out = summarize(rows, bootstrap_repeats=20)
        json.dumps(out, allow_nan=False)
        self.assertEqual(out['macro']['ik_fit_success_rate']['mean'], 0.)
        self.assertIsNone(out['macro']['mean_keypoint_error_mm']['mean'])
        self.assertIsNone(out['candidate_distributions']['direction_error_deg']['p50_p90_p95'])

    def test_object_weights_do_not_follow_candidate_count(self):
        a = dict(object_id='a', condition_id='x', decoded=False, fit_success=True,
                 direction_pass=False, finite_output=True)
        b = dict(a, object_id='b', fit_success=False)
        out = summarize([a]*8+[b]*2, bootstrap_repeats=20)
        self.assertEqual(out['macro']['ik_fit_success_rate']['mean'], .5)

    def test_roundtrip_and_nonfinite_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = hand_fixture(Path(tmp)); hand = URDFHand(path)
            k = hand.world_keypoints(np.eye(3), np.array([.3,-.2,.1]), np.zeros(22))
            row = decode_metrics(k, hand, [0,0,1])
            self.assertTrue(row['fit_success']); self.assertLess(row['max_keypoint_error_mm'], .01)
            invalid = decode_metrics(np.full((21,3),np.nan),hand,[0,0,1])
            self.assertFalse(invalid['finite_output']); self.assertFalse(invalid['fit_success'])

    def test_official_axis_override_is_in_memory_only(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path = hand_fixture(Path(tmp))
            spec = json.loads(path.read_text())
            spec['official_from_grasp_rotation'] = [[0,0,1],[1,0,0],[0,1,0]]
            path.write_text(json.dumps(spec)); before = path.read_bytes()
            hand = official_hand(path)
            self.assertEqual(path.read_bytes(), before)
            k = hand.world_keypoints(np.eye(3), np.zeros(3), np.zeros(22))
            row = decode_metrics(k, hand, [0,1,0])
            self.assertLess(row['direction_error_deg'], 1e-5)

    def test_comparison_rejects_noise_mismatch_and_incomplete_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); selection = root/'selection.json'
            write_json(selection, dict(rows=[{},{}]))
            keys = ['mean_keypoint_error_mm','mean_max_keypoint_error_mm','ik_fit_success_rate',
                    'direction_angular_error_deg','direction_pass_rate_10deg','joint_angle_diversity_rad',
                    'ik_valid_joint_angle_diversity_rad','decode_rate','finite_output_rate']
            folders = [root/'a',root/'b']
            for folder in folders:
                folder.mkdir(); np.save(folder/'noise.npy', np.zeros((2,8,21,3)))
                protocol = dict(selection=str(selection), selection_sha256=sha256(selection),
                                data_identity={}, seed=1, candidates=8, sampling_steps=20, sampler='Euler',
                                precision='BF16', ik={}, direction_definition={}, optimizer_steps=40000,
                                model_variant='object_only_h_less', checkpoint=str(folder/'step_040000.pt'))
                write_json(folder/'protocol.json',protocol)
                summary = dict(protocol_sha256=sha256(folder/'protocol.json'), candidates=16, objects=2,
                               macro={k:dict(mean=0.) for k in keys},
                               timing=dict(generation_ms_per_grasp=1.,ik_mean_ms=2.))
                write_json(folder/'summary.json',summary)
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(len(compare(folders,root/'valid')),2)
            np.save(folders[1]/'noise.npy',np.ones((2,8,21,3)))
            with self.assertRaisesRegex(ValueError,'identical'):
                compare(folders,root/'bad_noise')
            summary['candidates'] = 15
            write_json(folders[1]/'summary.json',summary)
            with self.assertRaisesRegex(ValueError,'incomplete'):
                compare(folders,root/'bad_count')


if __name__ == '__main__':
    unittest.main()
