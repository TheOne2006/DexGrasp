import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from dexgrasp.geometry.mesh import sample_surface, first_ray_hits, measure_palm_gap, inside_closed_mesh, validate_closed_mesh
from dexgrasp.geometry.hand import URDFHand, rotation_axis
from dexgrasp.geometry.decode import rigid_registration, decode_keypoints
from helpers import hand_fixture, box_mesh


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = hand_fixture(self.temp.name)
        self.hand = URDFHand(self.path)

    def test_area_sampling_distribution_and_triangle_bounds(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0], [12, 0, 0], [10, 2, 0]])
        points = sample_surface(vertices, np.array([[0, 1, 2], [3, 4, 5]]), 20000, np.random.default_rng(1))
        self.assertAlmostEqual(float((points[:, 0] > 5).mean()), .8, delta=.015)
        local = points[points[:, 0] < 5]
        self.assertTrue((local[:, :2] >= 0).all() and (local[:, :2].sum(-1) <= 1+1e-6).all())
        np.testing.assert_allclose(local.mean(0), [1/3, 1/3, 0], atol=.02)

    def test_rays_first_hit_gap_threshold_and_penetration(self):
        v, f = box_mesh()
        palm = self.hand.spec['palm']
        gap = measure_palm_gap(v, f, np.eye(3), np.zeros(3), palm)
        self.assertTrue(gap['valid'])
        self.assertAlmostEqual(gap['h'], .02)
        self.assertEqual(gap['hit_count'], 64)
        self.assertFalse(measure_palm_gap(v, f, np.eye(3), [0, 0, .04], palm)['valid'])
        missing = {**palm, 'ray_origins_m': [*palm['ray_origins_m'][:-1], [1, 1, 0]]}
        gap = measure_palm_gap(v, f, np.eye(3), [0, 0, 0], missing)
        self.assertFalse(gap['valid'])
        self.assertIsNone(gap['h'])
        self.assertEqual(gap['hit_count'], 63)
        np.testing.assert_array_equal(inside_closed_mesh([[0, 0, .1], [0, 0, -.1]], v, f), [True, False])
        with self.assertRaises(ValueError): validate_closed_mesh(v, f[:-1])

    def test_palm_gap_is_rigid_transform_invariant(self):
        v, f = box_mesh()
        rot, t = rotation_axis([0, 1, 0], .7), np.array([2, -3, .1])
        gap = measure_palm_gap(v@rot.T+t, f, rot, t, self.hand.spec['palm'])
        self.assertAlmostEqual(gap['h'], .02)

    def test_fk_analytic_jacobian_matches_finite_difference(self):
        q = np.linspace(-.1, .1, 22)
        _, jac = self.hand.fk(q, jacobian=True)
        for i in range(22):
            delta = np.eye(22)[i]*1e-6
            expected = (self.hand.fk(q+delta)-self.hand.fk(q-delta))/2e-6
            np.testing.assert_allclose(jac[:, :, i], expected, atol=1e-8)

    def test_svd_ik_roundtrip_and_proper_rotation(self):
        rotation, translation = rotation_axis([0, 0, 1], .6), np.array([.1, -.2, .3])
        q = np.zeros(22)
        target = self.hand.world_keypoints(rotation, translation, q)
        result = decode_keypoints(target, self.hand, rotation[:, 2], q_initial=q)
        self.assertTrue(result['ik_success'])
        np.testing.assert_allclose(result['R'], rotation, atol=1e-8)
        np.testing.assert_allclose(result['t'], translation, atol=1e-8)
        np.testing.assert_allclose(result['reconstructed_keypoints'], target, atol=1e-8)
        self.assertIsNone(result['physical_success'])
        a = np.random.default_rng(2).normal(size=(8, 3))
        r, _, err = rigid_registration(a, a*np.array([-1, 1, 1]))
        self.assertAlmostEqual(np.linalg.det(r), 1)
        self.assertGreater(err, .1)
        with self.assertRaises(ValueError): rigid_registration(np.zeros((3, 3)), np.zeros((3, 3)))

    def test_bad_calibration_and_moving_anchor_rejected(self):
        original = json.loads(self.path.read_text())
        invalid = {**original, 'calibrated': False}
        self.path.write_text(json.dumps(invalid))
        with self.assertRaises(ValueError): URDFHand(self.path)
        original['keypoints'][0]['link'] = 'link0'
        self.path.write_text(json.dumps(original))
        with self.assertRaises(ValueError): URDFHand(self.path)

    def test_unreachable_keypoints_are_not_ik_success(self):
        points = self.hand.fk(np.zeros(22))
        points[3:] += [0, 0, 3]
        result = decode_keypoints(points, self.hand, [0, 0, 1], max_nfev=5)
        self.assertFalse(result['ik_success'])
        self.assertTrue(result['joint_limits_valid'])

    def test_float32_joint_limit_roundoff_is_not_false_violation(self):
        self.hand.upper[:] = .3
        self.assertTrue(self.hand.in_limits(np.full(22, .3, np.float32)))
        self.assertFalse(self.hand.in_limits(np.full(22, .301, np.float32)))


if __name__ == '__main__':
    unittest.main()
