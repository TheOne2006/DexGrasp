import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from dexgrasp.test.physics import contact_metrics, pose_error, pca_diversity, PoseEvaluator


class PhysicsMetricsTests(unittest.TestCase):
    def test_signed_depth_cdc_and_separate_self_collision(self):
        result = contact_metrics([
            ('object', 'right_thumb_DP', -.003),
            ('right_thumb_PP', 'object', .001),
            ('right_index_DP', 'object', .001),
            ('right_middle_DP', 'right_ring_DP', -.004)])
        self.assertEqual(result['hand_object_penetration_mm'], 3)
        self.assertEqual(result['self_penetration_mm'], 4)
        self.assertAlmostEqual(result['contact_distance_consistency_mm'], 13)
        self.assertEqual(result['contact_link_count'], 2)
        self.assertAlmostEqual(result['contact_distance_mm'], 6.2)

    def test_no_contacts_is_not_success_and_has_censored_distances(self):
        result = contact_metrics([])
        self.assertEqual(result['hand_object_penetration_mm'], 0)
        self.assertEqual(result['contact_distance_mm'], 10)
        self.assertEqual(result['contact_distance_consistency_mm'], 0)
        self.assertEqual(result['contact_link_count'], 0)
        self.assertNotIn('single_pose_force_success', result)

    def test_palm_depth_counts_without_inventing_finger_contact(self):
        r = contact_metrics([('object','right_hand_C_MC',-.006)])
        self.assertEqual(r['hand_object_penetration_mm'],6)
        self.assertEqual(r['contact_link_count'],0)

    def test_mujoco_collision_distance_sign_and_mm(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="right_index_DP"><geom type="sphere" size=".01"/></body>
          <body name="object" pos=".017 0 0"><freejoint/><geom type="sphere" size=".01"/></body>
        </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        r = contact_metrics([(model.body(model.geom_bodyid[c.geom1]).name,
                              model.body(model.geom_bodyid[c.geom2]).name,c.dist) for c in data.contact])
        self.assertAlmostEqual(r['hand_object_penetration_mm'],3)
        self.assertEqual(r['self_penetration_mm'],0)

    def test_quaternion_sign_and_angle_units(self):
        q = Rotation.from_euler('z', 15, degrees=True).as_quat(scalar_first=True)
        p,a = pose_error([.03,.04,0],q,[0,0,0],[1,0,0,0])
        self.assertAlmostEqual(p,.05)
        self.assertAlmostEqual(a,15)
        self.assertEqual(pose_error([0,0,0],-q,[0,0,0],q),(0,0))

    def test_pca_collapse_and_rotational_invariance(self):
        self.assertIsNone(pca_diversity([]))
        self.assertIsNone(pca_diversity(np.zeros((8,28))))
        x = np.zeros((8,28)); x[:,0]=np.arange(8)
        self.assertAlmostEqual(pca_diversity(x)[0],100)
        rng = np.random.default_rng(1)
        x = rng.normal(size=(80,28)); q,_ = np.linalg.qr(rng.normal(size=(28,28)))
        np.testing.assert_allclose(pca_diversity(x),pca_diversity(x@q),atol=1e-10)

    def force_fixture(self):
        evaluator = PoseEvaluator.__new__(PoseEvaluator)
        evaluator.hand = SimpleNamespace(in_limits=lambda q:True)
        evaluator.model = object()
        evaluator.object_qadr = 0
        evaluator.object_id = 0
        evaluator.data = SimpleNamespace(qpos=np.array([0.,0.,0.,1.,0.,0.,0.]),
            qvel=np.zeros(6), warning=SimpleNamespace(number=np.zeros(8)),
            time=0., xfrc_applied=np.zeros((1,6)))
        def reset(*args, **kwargs):
            evaluator.data.time=0.
            evaluator.data.qpos[:]=[0,0,0,1,0,0,0]
            evaluator.data.xfrc_applied[:]=0
        evaluator.reset = Mock(side_effect=reset)
        evaluator.geometry = lambda:contact_metrics([])
        return evaluator

    def test_success_requires_six_independently_reset_force_directions(self):
        e = self.force_fixture()
        applied = []
        def step(model,data,nstep):
            data.time += .004*nstep
            if data.time > 1.01:
                applied.append(tuple(data.xfrc_applied[0,:3]))
        with patch('dexgrasp.test.physics.mujoco.mj_step',side_effect=step):
            result=e.evaluate(np.eye(3),np.zeros(3),np.zeros(22))
        self.assertTrue(result['single_pose_force_success'])
        self.assertEqual(len(set(applied)),6)
        self.assertEqual(e.reset.call_count,7)  # initial geometry plus each trial

    def test_one_direction_failure_cannot_be_hidden_by_later_resets(self):
        e = self.force_fixture()
        def step(model,data,nstep):
            data.time += .004*nstep
            if data.xfrc_applied[0,1] < 0:
                data.qpos[0]=.051
        with patch('dexgrasp.test.physics.mujoco.mj_step',side_effect=step):
            result=e.evaluate(np.eye(3),np.zeros(3),np.zeros(22))
        self.assertFalse(result['single_pose_force_success'])
        self.assertEqual(len(result['force_trials']),5)
        self.assertFalse(result['force_trials'][-1]['passed'])

    def test_severe_initial_penetration_is_counted_as_failure_without_simulation(self):
        e=self.force_fixture()
        e.geometry=lambda:contact_metrics([('object','right_index_DP',-.011)])
        with patch('dexgrasp.test.physics.mujoco.mj_step') as step:
            result=e.evaluate(np.eye(3),np.zeros(3),np.zeros(22))
        self.assertTrue(result['simulation_evaluated'])
        self.assertFalse(result['single_pose_force_success'])
        step.assert_not_called()


if __name__ == '__main__':
    unittest.main()
