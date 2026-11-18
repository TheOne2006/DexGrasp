import unittest

import numpy as np

from dexgrasp.data.dexonomy_scale import released_grasp_scale


class ReleasedScaleTest(unittest.TestCase):
    def test_candidates_in_one_scene_have_distinct_physical_scales(self):
        base = np.full(3, .05)
        np.testing.assert_allclose(released_grasp_scale(base, 2.6274824), [.13137412] * 3)
        np.testing.assert_allclose(released_grasp_scale(base, .8), [.04] * 3)
        np.testing.assert_array_equal(base, np.full(3, .05))

    def test_candidate_axis_cannot_be_treated_as_xyz(self):
        with self.assertRaises(ValueError):
            released_grasp_scale(.05, [1., 2., 3.])
