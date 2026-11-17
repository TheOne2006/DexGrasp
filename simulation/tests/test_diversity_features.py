"""Descriptor geometry and hand50 defaults independent of physical assets."""
import math
import unittest

import numpy as np

from dexonomy.diversity.features import FeatureConfig, build_features

class AnalyticHand:
    """Deterministic point map isolates descriptor geometry from asset/FK tests."""
    joint_names = tuple(range(22))
    joint_ranges = np.tile([-1., 1.], (22, 1))
    keypoint_weights = np.full(21, 1/21)
    keypoint_names = tuple(range(21))

    def keypoints(self, q):
        q = np.asarray(q)
        points = np.broadcast_to(np.arange(63).reshape(21, 3) / 1000,
                                 (len(q), 21, 3)).copy()
        points[:, :, 0] += q[:, :21] / 100
        points[:, :, 1] += q[:, 1:] / 100
        return points


def rotation(axis, angle):
    """Independent axis-angle construction for geometric test inputs."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * skew @ skew


class TestFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hand = AnalyticHand()

    def make_inputs(self):
        rng = np.random.default_rng(410)
        n = 3
        return {
            "root_translation": np.array([[0.14, -0.06, 0.2], [0.2, 0.1, 0.04], [-0.1, 0.2, 0.13]]),
            "root_rotation": np.stack([rotation([1, 2, 3], a) for a in (0.2, -0.6, 1.2)]),
            "q": rng.uniform(self.hand.joint_ranges[:, 0], self.hand.joint_ranges[:, 1], (n, 22)),
            "object_translation": np.array([[0.02, 0.03, 0.04], [0.05, -0.01, 0.06], [0.1, 0.0, -0.03]]),
            "object_rotation": np.stack([rotation([2, -1, 0.5], a) for a in (-0.7, 0.2, 1.1)]),
            "object_center": np.array([[0.01, -0.02, 0.005], [0.02, 0.01, -0.015], [-0.01, 0.005, 0.03]]),
            "object_radius": np.array([0.08, 0.12, 0.055]),
        }

    def features(self, inputs, config=None):
        return build_features(self.hand, **inputs, config=config)

    def test_common_world_transform_changes_no_block(self):
        inputs = self.make_inputs()
        original = self.features(inputs)
        global_R = rotation([-1, 3, 2], 1.72)
        global_t = np.array([3.0, -2.0, 0.7])
        transformed = dict(inputs)
        for key in ("root_translation", "object_translation"):
            transformed[key] = inputs[key] @ global_R.T + global_t
        for key in ("root_rotation", "object_rotation"):
            transformed[key] = global_R @ inputs[key]
        np.testing.assert_allclose(self.features(transformed), original, rtol=1e-12, atol=1e-13)
        self.assertEqual(original.shape, (3, 75))

    def test_palm_translation_changes_only_position_and_has_physical_scale(self):
        inputs = self.make_inputs()
        config = FeatureConfig(eta_t=0.4, w_t=2.5)
        baseline = self.features(inputs, config)
        displacement = np.array([0.014, -0.008, 0.006])
        changed = dict(inputs, root_translation=inputs["root_translation"] + displacement)
        difference = self.features(changed, config) - baseline
        np.testing.assert_array_equal(difference[:, 3:], 0)
        expected_squared_distance = config.w_t * np.dot(displacement, displacement) / (
            config.eta_t * inputs["object_radius"]) ** 2
        np.testing.assert_allclose(np.sum(difference[:, :3] ** 2, axis=1), expected_squared_distance)

    def test_palm_rotation_changes_only_orientation_with_chordal_formula(self):
        inputs = self.make_inputs()
        config = FeatureConfig(theta0=0.45, w_R=2.4)
        baseline = self.features(inputs, config)
        angles = np.array([1e-4, 0.9, math.pi])
        relative = np.stack([rotation([0.4, -2, 1], a) for a in angles])
        changed = dict(inputs, root_rotation=inputs["root_rotation"] @ relative)
        difference = self.features(changed, config) - baseline
        np.testing.assert_array_equal(difference[:, :3], 0)
        np.testing.assert_array_equal(difference[:, 12:], 0)
        expected = config.w_R * 4 * np.sin(angles / 2) ** 2 / config.theta0 ** 2
        np.testing.assert_allclose(np.sum(difference[:, 3:12] ** 2, axis=1), expected, rtol=1e-10)

    def test_finger_changes_only_hand_and_uses_weighted_rms_distance(self):
        inputs = self.make_inputs()
        config = FeatureConfig(ell_h=0.017, w_h=1.7)
        baseline = self.features(inputs, config)
        changed_q = inputs["q"].copy()
        changed_q[:, 7] += 0.2  # index PIP; palm remains exactly fixed
        changed = dict(inputs, q=changed_q)
        difference = self.features(changed, config) - baseline
        np.testing.assert_array_equal(difference[:, :12], 0)
        movement = self.hand.keypoints(changed_q) - self.hand.keypoints(inputs["q"])
        rms_squared = np.sum(movement ** 2 * self.hand.keypoint_weights[None, :, None], axis=(1, 2))
        self.assertTrue(np.all(rms_squared > 0))
        np.testing.assert_allclose(np.sum(difference[:, 12:] ** 2, axis=1),
                                   config.w_h * rms_squared / config.ell_h ** 2)
        np.testing.assert_allclose(rms_squared, np.sum(movement ** 2, axis=(1,2))/21)

    def test_empty_batch_dimension_and_fixed_hand50_defaults(self):
        from dexonomy.diversity.defaults import load_profile
        inputs = {key:value[:0] for key,value in self.make_inputs().items()}
        self.assertEqual(self.features(inputs).shape, (0,75))
        config, metadata = load_profile()
        self.assertEqual(config, FeatureConfig())
        np.testing.assert_array_equal([config.w_t, config.w_R, config.w_h],
            [.0026744040293024793, .022772101424186664, .23440514780267566])
        self.assertEqual(metadata['name'], 'hand50')
        self.assertEqual(metadata['reference']['calibration_keypoint_count'], 20)
        self.assertEqual(metadata['reference']['calibration_keypoint_scheme'],
                         'sharpa_segment_landmarks_v1')

    def test_relative_size_normalization_and_absolute_length_mode(self):
        inputs = self.make_inputs()
        baseline = self.features(inputs)
        larger_radius = dict(inputs, object_radius=2 * inputs["object_radius"])
        changed = self.features(larger_radius)
        np.testing.assert_allclose(changed[:, :3], baseline[:, :3] / 2)
        np.testing.assert_array_equal(changed[:, 3:], baseline[:, 3:])
        absolute = FeatureConfig(fixed_ell_t=0.03)
        np.testing.assert_array_equal(self.features(larger_radius, absolute),
                                      self.features(inputs, absolute))
        # Scale only object-space geometry and placement; the physical hand is unchanged.
        scaled = dict(inputs)
        for key in ("root_translation", "object_translation", "object_center", "object_radius"):
            scaled[key] = inputs[key] * 1.8
        np.testing.assert_allclose(self.features(scaled), baseline, atol=1e-13)

    def test_object_centroid_defines_translation_origin(self):
        inputs = self.make_inputs()
        center_world = inputs["object_translation"] + np.einsum(
            "nij,nj->ni", inputs["object_rotation"], inputs["object_center"])
        centered = dict(inputs, root_translation=center_world)
        np.testing.assert_allclose(self.features(centered)[:, :3], 0, atol=1e-14)

    def test_block_weights_and_resolution_ablation(self):
        inputs = self.make_inputs()
        baseline = self.features(inputs, FeatureConfig(w_t=1, w_R=1, w_h=1))
        weighted = self.features(inputs, FeatureConfig(w_t=4, w_R=0, w_h=9))
        np.testing.assert_allclose(weighted[:, :3], baseline[:, :3] * 2)
        np.testing.assert_array_equal(weighted[:, 3:12], 0)
        np.testing.assert_allclose(weighted[:, 12:], baseline[:, 12:] * 3)
        default = FeatureConfig()
        baseline = self.features(inputs, default)
        coarser = FeatureConfig(eta_t=2 * default.eta_t, theta0=2 * default.theta0,
                                ell_h=2 * default.ell_h)
        np.testing.assert_allclose(self.features(inputs, coarser), baseline / 2)

    def test_reject_invalid_geometry_and_calibration(self):
        inputs = self.make_inputs()
        for key, value in (("object_radius", np.zeros(3)),
                           ("root_rotation", np.tile(np.diag([1, 1, -1]), (3, 1, 1))),
                           ("object_center", np.full((3, 3), np.nan))):
            with self.assertRaises(ValueError):
                self.features(dict(inputs, **{key: value}))
        for config in ({"theta0": 0}, {"ell_h": -1}, {"eta_t": float("nan")},
                       {"w_t": -1}, {"w_t": 0, "w_R": 0, "w_h": 0}):
            with self.assertRaises(ValueError):
                FeatureConfig(**config)


if __name__ == "__main__":
    unittest.main()
