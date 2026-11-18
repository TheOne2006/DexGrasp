import unittest

import numpy as np

from dexgrasp.geometry.palm_depth import PalmSurfaceDepth


class PalmSurfaceDepthTest(unittest.TestCase):
    def test_rotated_palm_uses_nearest_triangle_interior(self):
        vertices = [[0, -1, -1], [0, 1, -1], [0, 0, 1]]
        mesh = PalmSurfaceDepth(vertices, [[0, 1, 2]], [1, 0.5, 0.5])
        rotations = np.stack([np.eye(3), np.diag([-1, -1, 1])])
        shifts = np.array([[0.1, 0, 0], [0.1, 0, 0]])
        np.testing.assert_allclose(mesh.measure(rotations, shifts), [0.126, 0.074], atol=1e-12)
