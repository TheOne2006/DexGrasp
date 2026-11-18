import unittest
from unittest.mock import Mock, patch

import numpy as np

from dexonomy.prediction_eval import floating_cpu


class ForceReferencePoseTest(unittest.TestCase):
    def test_reused_host_resets_before_planning_the_relative_force_test(self):
        initial_object_pose = np.array([0., 0., 0., 1., 0., 0., 0.])
        host = Mock()
        host.get_active_obj_pose.return_value = np.array([.21, .06, .09, 1., 0., 0., 0.])
        pregrasp = np.r_[initial_object_pose, np.zeros(22)]

        def reset(pose):
            np.testing.assert_array_equal(pose, pregrasp)
            host.get_active_obj_pose.return_value = initial_object_pose.copy()

        def plan(sim, scene, data, skip):
            np.testing.assert_array_equal(sim.get_active_obj_pose(), initial_object_pose)
            raise StopIteration

        host.reset_qpos.side_effect = reset
        with patch("dexonomy.op.eval_force_closure_warp._candidate_trajectory", side_effect=plan):
            with self.assertRaises(StopIteration):
                floating_cpu(host, {}, {"pregrasp_qpos": pregrasp[None]})
