"""The per-record scale convention used by released batched Dexonomy grasps."""

import numpy as np


def released_grasp_scale(base_scale, factor):
    """Return the metric scale after applying this candidate's augmentation."""
    base = np.asarray(base_scale, dtype=float)
    multiplier = np.asarray(factor, dtype=float)
    if base.shape not in ((), (3,)) or multiplier.shape != ():
        raise ValueError("Expected a scalar or three-axis scene scale and one candidate factor")
    if (
        not np.isfinite(base).all()
        or (base <= 0).any()
        or not np.isfinite(multiplier)
        or multiplier <= 0
    ):
        raise ValueError("Scene scale and candidate factor must be finite and positive")
    return base * multiplier
