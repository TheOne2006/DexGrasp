"""The formal Sharpa palm-surface-nearest-mean32-grid4y8z-m-v1 depth label."""

import numpy as np


class PalmSurfaceDepth:
    """Cache one scaled target mesh for exact Euclidean point-to-triangle queries."""

    def __init__(self, vertices, faces, scale):
        import trimesh

        self.mesh = trimesh.Trimesh(
            vertices=np.asarray(vertices) * np.asarray(scale), faces=faces, process=False
        )
        y = -0.025 + (np.arange(4) + 0.5) * 0.05 / 4
        z = 0.025 + (np.arange(8) + 0.5) * 0.05 / 8
        self.palm = np.column_stack((np.full(32, 0.026), np.repeat(y, 8), np.tile(z, 4)))

    def measure(self, official_rotation, translation):
        """Return one depth in metres per pose; translations use the scaled object frame."""
        rotation = np.asarray(official_rotation, dtype=float)
        translation = np.asarray(translation, dtype=float)
        if rotation.shape[-2:] != (3, 3) or translation.shape != rotation.shape[:-2] + (3,):
            raise ValueError("Expected matching rotations [...,3,3] and translations [...,3]")
        palm_world = self.palm @ rotation.swapaxes(-1, -2) + translation[..., None, :]
        from trimesh.proximity import nearby_faces
        from trimesh.triangles import closest_point

        points = palm_world.reshape(-1, 3)
        candidates = nearby_faces(self.mesh, points)
        counts = np.fromiter(map(len, candidates), dtype=np.int64, count=len(points))
        query = np.repeat(points, counts, axis=0)
        nearest = closest_point(self.mesh.triangles[np.concatenate(candidates)], query)
        distance = np.minimum.reduceat(
            np.linalg.norm(nearest - query, axis=1), np.r_[0, np.cumsum(counts)[:-1]]
        )
        return distance.reshape(rotation.shape[:-2] + (32,)).mean(axis=-1)
