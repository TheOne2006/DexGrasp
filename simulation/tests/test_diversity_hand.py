"""Independent CPU MuJoCo oracle tests for the batched Sharpa FK adapter."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from dexonomy.diversity.hand import (JOINT_NAMES, SharpaHand, DEFAULT_KEYPOINT_SCHEME,
                                 LEGACY_KEYPOINT_SCHEME, LANDMARK_RESOURCE_SHA256)


ASSET_ROOT = Path(os.environ["DEXGRASP_ASSET_ROOT"]) if os.environ.get("DEXGRASP_ASSET_ROOT") else None


@unittest.skipUnless(ASSET_ROOT is not None and (ASSET_ROOT / "hand/sharpa_v2/right_kp.xml").is_file(),
                     "set DEXGRASP_ASSET_ROOT to the Sharpa assets")
class TestSharpaHand(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hand = SharpaHand(ASSET_ROOT)

    def test_random_configurations_match_mujoco(self):
        hand = self.hand
        rng = np.random.default_rng(802)
        q = rng.uniform(hand.joint_ranges[:, 0], hand.joint_ranges[:, 1], (31, 22))
        actual = hand.keypoints(q, batch_size=7)
        # Compile and evaluate a separate model through MuJoCo's own FK.
        model = mujoco.MjModel.from_xml_path(str(hand.xml_path))
        data = mujoco.MjData(model)
        expected = np.empty_like(actual)
        for i, angles in enumerate(q):
            data.qpos[:] = angles
            mujoco.mj_kinematics(model, data)
            for p, (body, local) in enumerate(zip(hand.keypoint_bodies,
                                                   hand.keypoint_local_positions)):
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
                expected[i, p] = data.xmat[body_id].reshape(3, 3) @ local + data.xpos[body_id]
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=2e-14)
        np.testing.assert_allclose(hand.keypoints(q[0]), expected[0], atol=2e-14)

    def test_all_22_controls_are_observable(self):
        response = self.hand.joint_observability()
        self.assertEqual(tuple(response), JOINT_NAMES)
        self.assertGreater(min(response.values()), 1e-4)
        # At a generic configuration, simultaneous changes are locally observable too.
        baseline = self.hand.joint_ranges.mean(axis=1)
        delta = 1e-6
        changed = self.hand.keypoints(baseline + np.eye(22) * delta)
        jacobian = ((changed - self.hand.keypoints(baseline)) / delta).reshape(22, -1).T
        self.assertEqual(np.linalg.matrix_rank(jacobian, tol=1e-7), 22)

    def test_fixed_finger_weights_and_coverage(self):
        hand = self.hand
        self.assertEqual(len(hand.keypoint_names), 21)
        self.assertEqual(hand.keypoint_names[0], 'wrist')
        np.testing.assert_array_equal(hand.keypoint_weights, np.full(21, 1/21))
        np.testing.assert_allclose(hand.keypoint_weights[1:].reshape(5, 4).sum(axis=1), 4/21)
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            names = [n for n in hand.keypoint_names if n.startswith(finger + "_")]
            self.assertEqual(len(names), 4)
            self.assertIn(finger + "_tip", names)
        metadata = hand.metadata()
        self.assertEqual(metadata["palm_frame"], "right_hand_C_MC")
        self.assertEqual(len(metadata["xml_sha256"]), 64)
        self.assertEqual(metadata['keypoint_scheme'], DEFAULT_KEYPOINT_SCHEME)
        self.assertEqual(metadata['feature_dimension'], 75)
        self.assertEqual(metadata['landmark_definition']['sha256'], LANDMARK_RESOURCE_SHA256)
        self.assertEqual(hand.keypoints(np.empty((0, 22))).shape, (0, 21, 3))

    def test_rigid_anchors_and_pinky_cmc_point(self):
        q = np.zeros((2, 22)); q[1] = self.hand.joint_ranges.mean(axis=1)
        points = self.hand.keypoints(q)
        anchors = [0, 1, 5, 9, 13]
        np.testing.assert_array_equal(points[0, anchors], points[1, anchors])
        np.testing.assert_array_equal(points[:, 0], 0.)
        self.assertGreater(np.linalg.norm(points[0,17]-points[1,17]), 1e-4)

    def test_fixed_tip_offsets_are_not_legacy_pad_sites(self):
        expected = [[.027610999999627458,1.0142086610850226e-7,1.0142086610781804e-7],
                    [.026008999999649073,9.553639153294104e-8,9.553639153229653e-8],
                    [.0260092574262507,9.08108369409215e-17,9.08108369409215e-17],
                    [.02600925742589937,9.553733711235738e-8,9.553733711171285e-8],
                    [.026008999999649073,9.553639153294104e-8,9.553639153229653e-8]]
        np.testing.assert_allclose(self.hand.keypoint_local_positions[[4,8,12,16,20]], expected, rtol=0, atol=1e-15)
        self.assertFalse(np.allclose(expected, [.025,-.002,0]))

    def test_explicit_historical_scheme_remains_available(self):
        hand = SharpaHand(ASSET_ROOT, keypoint_scheme=LEGACY_KEYPOINT_SCHEME)
        self.assertEqual(hand.keypoints(np.zeros(22)).shape, (20,3))
        self.assertEqual(hand.metadata()['feature_dimension'], 72)
        self.assertEqual(hand.metadata()['keypoint_scheme'], 'sharpa_segment_landmarks_v1')
        self.assertEqual(hand.metadata()['selection_option'], LEGACY_KEYPOINT_SCHEME)
        np.testing.assert_array_equal(hand.keypoint_weights, np.full(20, .05))
        self.assertIn('thumb_pad', hand.keypoint_names)
        with self.assertRaisesRegex(ValueError, 'Unknown Sharpa keypoint scheme'):
            SharpaHand(ASSET_ROOT, keypoint_scheme='invented')

    def test_rejects_changed_joint_mapping(self):
        model = mujoco.MjModel.from_xml_path(str(self.hand.xml_path))
        model.jnt_axis[3] = [1,0,0]
        with patch('dexonomy.diversity.hand.mujoco.MjModel.from_xml_path', return_value=model):
            with self.assertRaisesRegex(ValueError, 'native kinematics differs'):
                SharpaHand(ASSET_ROOT)

    def test_rejects_modified_bundled_definition(self):
        with patch('dexonomy.diversity.hand.LANDMARK_RESOURCE_SHA256', '0'*64):
            with self.assertRaisesRegex(ValueError, 'definition SHA256 mismatch'):
                SharpaHand(ASSET_ROOT)

    def test_reject_bad_joint_arrays(self):
        for bad in (np.zeros((1, 29)), np.zeros((2, 2, 22)), np.full((22,), np.nan)):
            with self.assertRaises(ValueError):
                self.hand.keypoints(bad)
        with self.assertRaises(ValueError):
            self.hand.keypoints(np.zeros(22), batch_size=0)


if __name__ == "__main__":
    unittest.main()
