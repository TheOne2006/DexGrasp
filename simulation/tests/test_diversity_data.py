"""Trusted native fixtures: no external dataset or hand assets required."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dexonomy.diversity.data import clear_caches, discover_grasps, load_grasp


class NativeDataTests(unittest.TestCase):
    def setUp(self):
        clear_caches()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.assets = self.root / 'assets'
        self.mesh = self.assets / 'object/DGN_5k/processed_data/test_object/mesh/simplified.obj'
        self.mesh.parent.mkdir(parents=True)
        # A right triangle has area centroid (1/3,1/3,0), E||X-c||²=1/9.
        self.mesh.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
        self.scene_path = self.root / 'scene.npy'
        self.scene = dict(scene_id='test_object_floating_scale010', task=dict(
            type='force_closure', obj_name='test_object'), scene={'test_object': dict(
                type='rigid_object', file_path=str(self.mesh), scale=np.array([.1]*3),
                pose=np.array([0., 0., 0., 1., 0., 0., 0.]))})
        self.save_scene()
        self.data = dict(hand_name='sharpa_v2', tmpl_name='family_a',
                         grasp_qpos=np.array([[.01, .02, .03, 1., 0., 0., 0.] + [0.]*22]),
                         obj_pose=np.array([0., 0., 0., 1., 0., 0., 0.]),
                         obj_scale=np.array([.1]*3, dtype=np.float32),
                         scene_path=str(self.scene_path), pipeline_stage='stable_grasp',
                         stability_metrics={'passes': True})

    def tearDown(self):
        clear_caches()
        self.temporary.cleanup()

    def save_scene(self):
        np.save(self.scene_path, self.scene)

    def write(self, name='grasp.npy', data=None):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.data if data is None else data)
        return path

    def load(self, path, **kwargs):
        return load_grasp(path, asset_root=self.assets, **kwargs)

    def test_native_pose_and_surface_geometry(self):
        record = self.load(self.write())
        self.assertEqual(record.protocol, 'floating')
        self.assertEqual(record.q.shape, (22,))
        np.testing.assert_allclose(record.root_translation, [.01, .02, .03])
        np.testing.assert_allclose(record.root_rotation, np.eye(3))
        np.testing.assert_allclose(record.object_center, [.1/3, .1/3, 0])
        self.assertAlmostEqual(record.object_radius, .1/3)
        self.assertEqual(record.scale, (.1, .1, .1))
        self.assertEqual(record.scene_path, self.scene_path)
        self.assertEqual(record.mesh_path, self.mesh)

    def test_quaternion_sign_is_equivalent(self):
        a = self.load(self.write('a.npy'))
        self.data['grasp_qpos'][0, 3:7] *= -1
        self.data['obj_pose'][3:] *= -1
        b = self.load(self.write('b.npy'))
        np.testing.assert_array_equal(a.root_rotation, b.root_rotation)
        np.testing.assert_array_equal(a.object_rotation, b.object_rotation)

    def test_template_family_does_not_split_group(self):
        a = self.load(self.write('a.npy'))
        self.data['tmpl_name'] = 'family_b'
        b = self.load(self.write('b.npy'))
        self.assertEqual(a.group_key, b.group_key)

    def test_candidate_identity_keeps_namespace_and_checks_both_provenances(self):
        identity = 'dataset-rank000/job/fingerprint'
        self.data['provenance'] = {'candidate_id': identity}
        self.data['experiment_provenance'] = {'candidate_id': identity}
        self.assertEqual(self.load(self.write()).candidate_identity, identity)
        self.data.pop('provenance')
        self.assertEqual(self.load(self.write('fallback.npy')).candidate_identity, identity)
        self.data['provenance'] = {'candidate_id': 'different-dataset/job/fingerprint'}
        with self.assertRaisesRegex(ValueError, 'conflicting candidate identities'):
            self.load(self.write('conflict.npy'))
        self.data.pop('provenance')
        self.data.pop('experiment_provenance')
        self.assertIsNone(self.load(self.write('without_id.npy')).candidate_identity)

    def test_candidate_identity_is_not_coerced_from_invalid_values(self):
        for value in ('', ' ', 14, ['candidate']):
            with self.subTest(value=value):
                self.data['provenance'] = {'candidate_id': value}
                with self.assertRaisesRegex(ValueError, 'candidate_id must be a nonempty string'):
                    self.load(self.write())

    def test_success_policy_does_not_silently_admit_unknown(self):
        self.data.pop('stability_metrics')
        self.data.pop('pipeline_stage')
        path = self.write()
        with self.assertRaisesRegex(ValueError, 'success is unknown'):
            self.load(path)
        self.assertIsNotNone(self.load(path, selection='all'))
        self.assertIsNotNone(self.load(self.write('success/example.npy')))

    def test_explicit_failure_overrides_success_directory(self):
        self.data['stability_metrics']['passes'] = False
        path = self.write('success/example.npy')
        self.assertIsNone(self.load(path))
        self.assertIsNotNone(self.load(path, selection='all'))

    def test_nonboolean_success_rejected(self):
        self.data['stability_metrics']['passes'] = 'False'
        with self.assertRaisesRegex(ValueError, 'must be a boolean'):
            self.load(self.write())

    def test_discovery_prefers_success_over_stage_copies(self):
        a = self.write('run/rank-000/success/a.npy')
        b = self.write('run/rank-001/success/b.npy')
        self.write('run/jobs/init.npy', {'entries': []})
        self.write('run/rank-000/jobs/eval.npy', {'results': []})
        self.assertEqual(discover_grasps([self.root / 'run', a]), sorted([a, b]))

    def test_discovery_empty_and_unknown_format_fail(self):
        empty = self.root / 'empty'
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, 'No native grasp'):
            discover_grasps(empty)
        path = self.write(data={'entries': []})
        with self.assertRaisesRegex(ValueError, 'not a native grasp'):
            self.load(path)

    def test_relative_scene_and_mesh_paths(self):
        self.scene['scene']['test_object']['file_path'] = str(self.mesh.relative_to(self.root))
        self.save_scene()
        self.data['scene_path'] = 'scene.npy'
        self.assertEqual(self.load(self.write()).mesh_path, self.mesh)

    def test_old_absolute_asset_paths_are_remapped(self):
        self.scene['scene']['test_object']['file_path'] = (
            '/old/workspace/Dexonomy/assets/object/DGN_5k/processed_data/test_object/mesh/simplified.obj')
        self.save_scene()
        self.assertEqual(self.load(self.write()).mesh_path, self.mesh)

    def test_old_scene_asset_path_is_remapped(self):
        destination = self.assets / 'object/DGN_5k/scene_cfg/scale010.npy'
        destination.parent.mkdir(parents=True)
        np.save(destination, self.scene)
        self.data['scene_path'] = '/old/Dexonomy/assets/object/DGN_5k/scene_cfg/scale010.npy'
        self.assertEqual(self.load(self.write()).scene_path, destination)

    def test_mismatched_scale_and_pose_fail(self):
        self.data['obj_scale'] = [.2]*3
        with self.assertRaisesRegex(ValueError, 'obj_scale differs'):
            self.load(self.write('wrong_scale.npy'))
        self.data['obj_scale'] = [.1]*3
        self.data['obj_pose'][0] = 1.
        with self.assertRaisesRegex(ValueError, 'obj_pose differs'):
            self.load(self.write('wrong_pose.npy'))

    def test_bad_quaternion_and_wrong_hand_fail(self):
        self.data['grasp_qpos'][0, 3:7] = 0
        with self.assertRaisesRegex(ValueError, 'unit MuJoCo'):
            self.load(self.write('bad_q.npy'))
        self.data['hand_name'] = 'other_hand'
        with self.assertRaisesRegex(ValueError, 'unsupported hand_name'):
            self.load(self.write('bad_hand.npy'))

    def test_nonrigid_object_fails(self):
        self.scene['scene']['test_object']['type'] = 'articulated_object'
        self.save_scene()
        with self.assertRaisesRegex(ValueError, 'only rigid_object'):
            self.load(self.write())

    def test_protocol_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, 'contradicts native protocol'):
            self.load(self.write(), protocol='tabletop')

    def test_tabletop_uses_initial_pose_and_separates_stable_poses(self):
        self.data.pop('stability_metrics')
        self.data['pipeline_stage'] = 'stability_candidate'
        self.data['tabletop_metrics'] = {'joint_success': True, 'protocol': 'tabletop-slide-fc/v1'}
        # Terminal object pose must not be paired with pre-Slide grasp_qpos.
        self.data['tabletop_actual_object_pose'] = np.array([0, 0, .1, 1, 0, 0, 0])
        self.scene['scene_id'] = 'test_object_tabletop_pose000_scale010'
        self.scene['task'].update(type='slide', axis=[0, 0, 1], distance=.1)
        self.scene['scene']['table'] = dict(type='plane', pose=[0, 0, 0, 1, 0, 0, 0],
                                           size=[0, 0, 1])
        self.save_scene()
        a = self.load(self.write('a.npy'))
        self.assertEqual(a.protocol, 'tabletop')
        np.testing.assert_array_equal(a.object_translation, [0, 0, 0])
        self.scene['scene_id'] = 'test_object_tabletop_pose001_scale010'
        second_scene = self.root / 'scene_second.npy'
        np.save(second_scene, self.scene)
        self.data['scene_path'] = str(second_scene)
        b = self.load(self.write('b.npy'))
        self.assertNotEqual(a.group_key, b.group_key)

    def test_scaled_geometry_and_float_precision_grouping(self):
        a = self.load(self.write('a.npy'))
        self.scene['scene']['test_object']['scale'] = np.array([.2]*3, dtype=np.float32)
        second_scene = self.root / 'scene_second.npy'
        np.save(second_scene, self.scene)
        self.data['scene_path'] = str(second_scene)
        self.data['obj_scale'] = [.2]*3
        b = self.load(self.write('b.npy'))
        np.testing.assert_allclose(b.object_center, a.object_center * 2)
        self.assertAlmostEqual(b.object_radius, a.object_radius * 2)
        self.assertNotEqual(a.group_key, b.group_key)
        self.scene['scene']['test_object']['scale'] = [.2]*3
        third_scene = self.root / 'scene_third.npy'
        np.save(third_scene, self.scene)
        self.data['scene_path'] = str(third_scene)
        c = self.load(self.write('c.npy'))
        self.assertEqual(b.group_key, c.group_key)


    def test_clear_caches_observes_replaced_scene_and_mesh(self):
        path = self.write()
        a = self.load(path)
        self.mesh.write_text('v 0 0 0\nv 2 0 0\nv 0 2 0\nf 1 2 3\n')
        self.scene['scene']['test_object']['scale'] = [.2]*3
        self.save_scene()
        self.data['obj_scale'] = [.2]*3
        self.write()
        clear_caches()
        b = self.load(path)
        self.assertAlmostEqual(b.object_radius, a.object_radius * 4)
        self.assertEqual(b.scale, (.2, .2, .2))

    def _make_tabletop(self):
        self.data.pop('stability_metrics', None)
        self.data['pipeline_stage'] = 'stability_candidate'
        self.data['tabletop_metrics'] = {'joint_success': True}
        self.scene['scene_id'] = 'same_tabletop_scene_id'
        self.scene['task'].update(type='slide', axis=[0., 0., 1.], distance=.1)
        self.scene['scene']['table'] = dict(type='plane', pose=[0., 0., 0., 1., 0., 0., 0.],
                                           size=[0., 0., 1.])
        self.save_scene()

    def _write_new_scene(self, name):
        scene_path = self.root / f'{name}_scene.npy'
        np.save(scene_path, self.scene)
        self.data['scene_path'] = str(scene_path)
        return self.write(f'{name}.npy')

    def test_same_scene_id_different_support_geometry_does_not_merge(self):
        self._make_tabletop()
        a = self.load(self.write('a.npy'))
        self.scene['scene']['test_object']['pose'][2] = .03
        self.data['obj_pose'][2] = .03
        moved_object = self.load(self._write_new_scene('moved_object'))
        self.assertNotEqual(a.group_key, moved_object.group_key)
        self.scene['scene']['test_object']['pose'][2] = 0
        self.data['obj_pose'][2] = 0
        self.scene['scene']['table']['pose'][2] = .02
        moved_table = self.load(self._write_new_scene('moved_table'))
        self.assertNotEqual(a.group_key, moved_table.group_key)
        self.scene['scene']['table']['pose'][2] = 0
        self.scene['scene']['table']['size'] = [.2, .3, .01]
        resized_table = self.load(self._write_new_scene('resized_table'))
        self.assertNotEqual(a.group_key, resized_table.group_key)

    def test_tabletop_float_precision_and_quaternion_sign_do_not_split(self):
        self._make_tabletop()
        self.scene['scene']['test_object']['pose'] = np.array(
            [0., 0., .0317, .9238795325112867, .3826834323650898, 0., 0.])
        self.data['obj_pose'] = self.scene['scene']['test_object']['pose'].copy()
        self.scene['scene']['table']['size'] = [.1234, .4567, .01]
        self.save_scene()
        a = self.load(self.write('a.npy'))
        for obj in self.scene['scene'].values():
            for field in ('pose', 'scale', 'size'):
                if field in obj:
                    obj[field] = np.asarray(obj[field], dtype=np.float32)
            obj['pose'][3:] *= -1
        self.scene['task']['distance'] = np.float32(.1)
        self.scene['task']['axis'] = np.asarray([0, 0, 1], dtype=np.float32)
        self.data['obj_pose'] = self.scene['scene']['test_object']['pose'].copy()
        b = self.load(self._write_new_scene('float32'))
        self.assertEqual(a.group_key, b.group_key)

    def test_tabletop_common_world_transform_preserves_geometry_signature(self):
        self._make_tabletop()
        a = self.load(self.write('a.npy'))
        # Rotate every scene frame by +90 degrees around Z and translate it.
        quaternion = [np.sqrt(.5), 0., 0., np.sqrt(.5)]
        for obj in self.scene['scene'].values():
            obj['pose'] = np.array([.1, -.2, .3] + quaternion)
        self.data['obj_pose'] = self.scene['scene']['test_object']['pose'].copy()
        b = self.load(self._write_new_scene('rigid_transform'))
        self.assertEqual(a.group_key, b.group_key)

    def test_tabletop_extra_environment_objects_are_not_ignored(self):
        self._make_tabletop()
        self.scene['scene']['obstacle'] = dict(type='rigid_object')
        self.save_scene()
        with self.assertRaisesRegex(ValueError, 'other environment objects are unsupported'):
            self.load(self.write())


if __name__ == '__main__':
    unittest.main()
