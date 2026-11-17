"""Global native API/CLI contracts with temporary trusted generation fixtures."""
from contextlib import redirect_stderr
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dexonomy.diversity import evaluate_dataset, evaluate_pair_features
from dexonomy.diversity import evaluator
from dexonomy.diversity.data import clear_caches
from dexonomy.diversity.features import FeatureConfig
from dexonomy.diversity.hand import DEFAULT_KEYPOINT_SCHEME


class FixtureHand:
    """Only replaces asset-dependent FK; native parsing/features stay real."""
    joint_names = tuple(range(22))
    keypoint_names = tuple(range(21))
    keypoint_weights = np.full(21, 1/21)

    def keypoints(self, q):
        result = np.zeros((len(q), 21, 3))
        result[:, :, 0] = np.asarray(q)[:, :21] / 100
        return result

    def metadata(self):
        return {'keypoint_count': 21, 'feature_dimension': 75,
                'keypoint_scheme': DEFAULT_KEYPOINT_SCHEME,
                'palm_frame': 'right_hand_C_MC', 'test_fixture_fk': True}


class NativeFixtures:
    def setUp(self):
        clear_caches()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(clear_caches)
        self.root = Path(self.temp.name)
        self.success = self.root / 'run/success'
        self.success.mkdir(parents=True)
        self.assets = self.root / 'assets'
        self.assets.mkdir()
        self.mesh = self.assets / 'triangle.obj'
        self.mesh.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')

    def write_native(self, name, x=0., *, obj='object_a', scale=.1,
                     candidate=None, passes=True, identify=True):
        scene_path = self.root / f'{name}_scene.npy'
        pose = np.array([0., 0., 0., 1., 0., 0., 0.])
        np.save(scene_path, {'scene_id': f'{obj}_scale{scale}',
            'task': {'type': 'force_closure', 'obj_name': obj},
            'scene': {obj: {'type': 'rigid_object', 'file_path': str(self.mesh),
                           'pose': pose, 'scale': [scale] * 3}}})
        data = {'hand_name': 'sharpa_v2', 'tmpl_name': 'fixture_family',
                'grasp_qpos': np.array([[scale/3+x, scale/3, 0., 1., 0., 0., 0.] + [0.] * 22]),
                'obj_pose': pose, 'obj_scale': [scale] * 3,
                'scene_path': str(scene_path), 'pipeline_stage': 'stable_grasp',
                'stability_metrics': {'passes': passes}}
        if identify:
            data['provenance'] = {'candidate_id': candidate or 'fixture/native/' + name}
        path = self.success / f'{name}.npy'
        np.save(path, data)
        return path

    def evaluate(self, inputs=None, **kwargs):
        with patch.object(evaluator, 'SharpaHand', return_value=FixtureHand()):
            return evaluate_dataset(self.success if inputs is None else inputs,
                                    asset_root=self.assets, **kwargs)


class NativeGlobalTests(NativeFixtures, unittest.TestCase):
    def test_all_singleton_conditions_cross_object_and_scale_pairs_are_included(self):
        self.write_native('a', 0)
        self.write_native('b', 1, scale=.2)
        self.write_native('c', 2, obj='object_b')
        config = FeatureConfig(fixed_ell_t=1., w_t=1., w_R=0., w_h=0.)
        result = self.evaluate(config=config, feature_batch_size=2)
        self.assertEqual((result['total_N'], result['total_U'], result['total_B']), (3, 3, 3))
        self.assertEqual(result['counts']['singleton_condition_groups'], 3)
        self.assertAlmostEqual(result['dataset_collision_mean'], (2*math.exp(-1)+math.exp(-4))/3)
        self.assertEqual(result['pair_population']['cross_condition_pairs'], 3)
        self.assertEqual(result['pair_population']['cross_object_pairs'], 2)
        self.assertEqual(result['pair_population']['different_scale_value_pairs'], 2)
        self.assertEqual(result['kernel_profile']['name'], 'custom')

    def test_relabeling_object_group_and_template_does_not_change_global_weighting(self):
        a = self.write_native('a', .01)
        b = self.write_native('b', .02)
        before = self.evaluate()
        # Identity/labels change, all physical arrays and native mesh stay fixed.
        data = np.load(b, allow_pickle=True).item()
        data['tmpl_name'] = 'unrelated_label'
        scene_path = Path(data['scene_path'])
        scene = np.load(scene_path, allow_pickle=True).item()
        scene['task']['obj_name'] = 'object_b'
        scene['scene']['object_b'] = scene['scene'].pop('object_a')
        scene['scene_id'] = 'object_b_scale0.1'
        np.save(scene_path, scene); np.save(b, data); clear_caches()
        after = self.evaluate([a, b])
        self.assertEqual(before['dataset_diversity'], after['dataset_diversity'])
        self.assertEqual(after['counts']['num_objects'], 2)

    def test_duplicate_native_identity_reports_both_sources(self):
        a = self.write_native('a', candidate='namespace/job/duplicate')
        b = self.write_native('b', candidate='namespace/job/duplicate')
        with self.assertRaisesRegex(ValueError, 'Duplicate candidate identity') as error:
            self.evaluate()
        self.assertIn(str(a), str(error.exception))
        self.assertIn(str(b), str(error.exception))

    def test_equal_poses_with_distinct_ids_and_namespaces_are_retained(self):
        self.write_native('a', candidate='dataset_a/job/fingerprint')
        self.write_native('b', candidate='dataset_b/job/fingerprint')
        result = self.evaluate()
        self.assertEqual(result['total_N'], 2)
        self.assertEqual(result['dataset_diversity'], 0.)

    def test_paths_deduplicate_but_missing_id_copies_remain_records(self):
        a = self.write_native('a', identify=False)
        b = self.write_native('b', identify=False)
        result = self.evaluate([self.success, a, b, a])
        self.assertEqual(result['total_N'], 2)
        self.assertEqual(result['total_B'], 1)

    def test_selection_is_applied_before_candidate_uniqueness(self):
        self.write_native('included', candidate='same')
        self.write_native('failed_copy', candidate='same', passes=False)
        result = self.evaluate()
        self.assertEqual(result['total_N'], 1)
        self.assertEqual(result['counts']['excluded_by_selection'], 1)
        self.assertIsNone(result['dataset_diversity'])
        with self.assertRaisesRegex(ValueError, 'Duplicate candidate identity'):
            self.evaluate(selection='all')

    def test_empty_and_all_filtered_do_not_require_hand_assets(self):
        for make_failure in (False, True):
            if make_failure:
                self.write_native('failed', passes=False)
            with patch.object(evaluator, 'SharpaHand', side_effect=AssertionError('no FK needed')):
                result = evaluate_dataset(self.success, asset_root=self.assets)
            self.assertEqual(result['total_N'], 0)
            self.assertEqual(result['total_B'], 0)
            self.assertEqual(result['feature_dimension'], 75)
            self.assertIsNone(result['dataset_diversity'])

    def test_default_profile_metadata_and_weighted_cache_match_metric(self):
        self.write_native('a', .01)
        self.write_native('b', .02, obj='object_b')
        destination = self.root / 'weighted.npy'
        result = self.evaluate(save_features=destination, feature_batch_size=1)
        self.assertEqual(result['schema'], 'dexonomy-global-pair-diversity/v1')
        self.assertEqual(result['pair_scope'], 'global')
        profile = result['kernel_profile']
        self.assertEqual(profile['name'], 'hand50')
        self.assertEqual(profile['reference']['calibration_keypoint_count'], 20)
        self.assertEqual(profile['current_keypoint_count'], 21)
        self.assertTrue(profile['not_recalibrated_for_current_keypoints'])
        saved = np.load(destination, allow_pickle=False)
        self.assertEqual(saved.shape, (2, 75))
        self.assertTrue(result['saved_global_features']['weights_already_applied'])
        self.assertEqual(result['saved_global_features']['sha256'], hashlib.sha256(destination.read_bytes()).hexdigest())
        self.assertEqual(evaluate_pair_features(saved)['dataset_diversity'], result['dataset_diversity'])
        with self.assertRaises(ValueError):
            self.evaluate(save_features=destination)

    def test_native_global_sample_budget_applies_to_all_records(self):
        self.write_native('a')
        self.write_native('b', .01, obj='object_b')
        self.write_native('c', .02, scale=.2)
        result = self.evaluate(mode='sample', pairs_per_grasp=7, seed=14, batch_size=4)
        self.assertEqual((result['total_N'], result['total_U'], result['total_B']), (3, 3, 21))
        explicit = self.evaluate(mode='sample', pairs_per_grasp=7, num_pairs=53)
        self.assertEqual(explicit['total_B'], 53)

    def test_cli_json_sample_and_no_overwrite(self):
        from dexonomy.diversity.__main__ import main
        self.write_native('a'); self.write_native('b', .01)
        output = self.root / 'result.json'
        with patch.object(evaluator, 'SharpaHand', return_value=FixtureHand()):
            main([str(self.success), '--asset-root', str(self.assets), '--mode', 'sample',
                  '--num-pairs', '19', '--seed', '4', '--joint-only', '--quiet', '--output', str(output)])
        result = json.loads(output.read_text())
        self.assertEqual(result['total_B'], 19)
        self.assertEqual(list(result['scores']), ['joint'])
        before = output.read_bytes()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main([str(self.success), '--asset-root', str(self.assets), '--output', str(output)])
        self.assertEqual(output.read_bytes(), before)

    def test_cli_subprocess_import_is_cpu_only_and_works_without_external_assets(self):
        code = ('import sys; import dexonomy.diversity; '
                'assert "torch" not in sys.modules; assert "warp" not in sys.modules')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        subprocess.run([sys.executable, '-c', code], check=True, env=env, capture_output=True, text=True)
        completed = subprocess.run([sys.executable, '-m', 'dexonomy.diversity', str(self.success),
                                    '--asset-root', str(self.assets), '--quiet'],
                                   check=True, env=env, capture_output=True, text=True)
        result = json.loads(completed.stdout)
        self.assertEqual(result['total_N'], 0)
        self.assertEqual(result['feature_dimension'], 75)
        self.assertEqual(result['kernel_profile']['name'], 'hand50')


ASSET_ROOT = Path(os.environ['DEXGRASP_ASSET_ROOT']) if os.environ.get('DEXGRASP_ASSET_ROOT') else None


@unittest.skipUnless(ASSET_ROOT is not None and (ASSET_ROOT / 'hand/sharpa_v2/right_kp.xml').is_file(),
                     'set DEXGRASP_ASSET_ROOT to enable real Sharpa native CLI smoke')
class NativeRealAssetSmoke(NativeFixtures, unittest.TestCase):
    def test_real_native_cli_75d_matches_direct_saved_features(self):
        self.write_native('a'); self.write_native('b', .01, obj='object_b')
        self.write_native('c', .02, scale=.2)
        feature_path = self.root / 'features.npy'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        completed = subprocess.run([sys.executable, '-m', 'dexonomy.diversity', str(self.success),
            '--asset-root', str(ASSET_ROOT), '--save-features', str(feature_path), '--quiet'],
            check=True, env=env, capture_output=True, text=True)
        result = json.loads(completed.stdout)
        self.assertEqual((result['total_N'], result['total_U']), (3, 3))
        self.assertEqual(result['hand']['keypoint_scheme'], DEFAULT_KEYPOINT_SCHEME)
        self.assertEqual(result['hand']['keypoint_count'], 21)
        saved = np.load(feature_path, allow_pickle=False)
        self.assertEqual(saved.shape, (3, 75))
        for block, sl in [('joint', slice(None)), ('translation', slice(3)),
                          ('rotation', slice(3, 12)), ('hand', slice(12, None))]:
            mu = math.fsum(math.exp(-float(np.sum((saved[i, sl]-saved[j, sl])**2)))
                           for i, j in [(0, 1), (0, 2), (1, 2)]) / 3
            self.assertAlmostEqual(result['scores'][block]['mu'], mu, places=14)


if __name__ == '__main__':
    unittest.main()
