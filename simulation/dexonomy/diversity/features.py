"""Object-relative palm pose and palm-local Sharpa landmark descriptors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    # Fixed hand50 parameters inherited from the historical 20-point reference.
    # They are not refitted to the current 21-point representation or inputs.
    eta_t: float = 0.5
    theta0: float = math.pi / 6
    ell_h: float = 0.02
    fixed_ell_t: float | None = None
    w_t: float = 0.0026744040293024793
    w_R: float = 0.022772101424186664
    w_h: float = 0.23440514780267566

    def __post_init__(self):
        for name in ("eta_t", "theta0", "ell_h"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.fixed_ell_t is not None and (
            not math.isfinite(self.fixed_ell_t) or self.fixed_ell_t <= 0
        ):
            raise ValueError("fixed_ell_t must be finite and positive")
        for name in ("w_t", "w_R", "w_h"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.w_t + self.w_R + self.w_h == 0:
            raise ValueError("at least one block weight must be positive")

    def metadata(self):
        return {
            **asdict(self),
            "translation_scale_mode": "absolute" if self.fixed_ell_t else "object_relative",
            "length_unit": "m",
            "angle_unit": "rad",
            "calibration": "configuration_only; provenance supplied by dataset evaluator",
        }


def _array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _rotation(value, n, name):
    result = _array(value, (n, 3, 3), name)
    if not np.allclose(result.swapaxes(1, 2) @ result, np.eye(3), atol=1e-6, rtol=0) or not np.allclose(
        np.linalg.det(result), 1.0, atol=1e-6, rtol=0
    ):
        raise ValueError(f"{name} must contain proper rotation matrices")
    return result


def build_features(
    hand,
    root_translation,
    root_rotation,
    q,
    object_translation,
    object_rotation,
    object_center,
    object_radius,
    config: FeatureConfig | None = None,
):
    """Return float64 (N,12+3*L) z (default L=21, dimension 75).

    All positions are in metres, angles radians. Equal point weights include
    the wrist and rigid palm anchors; outer kernel weights are not refitted
    when the landmark representation changes.

    object_center is the scaled, object-local surface centroid. object_translation
    and object_rotation describe the native mesh origin in the world. Sharpa's
    generated official root frame is its palm frame, so no extra wrist transform
    is required. Object scale affects center/radius, never the hand pose or FK.
    """
    config = config or FeatureConfig()
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != len(hand.joint_names) or not np.isfinite(q).all():
        raise ValueError(f"q must be finite with shape (N,{len(hand.joint_names)})")
    n = len(q)
    t = _array(root_translation, (n, 3), "root_translation")
    R = _rotation(root_rotation, n, "root_rotation")
    t_obj = _array(object_translation, (n, 3), "object_translation")
    R_obj = _rotation(object_rotation, n, "object_rotation")
    center = _array(object_center, (n, 3), "object_center")
    radius = _array(object_radius, (n,), "object_radius")
    if np.any(radius <= 0):
        raise ValueError("object_radius must be positive")
    inverse = R_obj.swapaxes(1, 2)
    relative_t = np.einsum("nij,nj->ni", inverse, t - t_obj) - center
    relative_R = inverse @ R
    ell_t = config.fixed_ell_t if config.fixed_ell_t is not None else config.eta_t * radius[:, None]
    local_points = hand.keypoints(q)
    weights = np.asarray(hand.keypoint_weights, dtype=np.float64)
    z = np.concatenate([
        math.sqrt(config.w_t) * relative_t / ell_t,
        math.sqrt(config.w_R) * relative_R.reshape(n, 9) / (math.sqrt(2) * config.theta0),
        (math.sqrt(config.w_h) * local_points * np.sqrt(weights)[None, :, None] / config.ell_h).reshape(n, 3 * len(weights)),
    ], axis=1)
    if not np.isfinite(z).all():
        raise ValueError("descriptor overflow; check feature scales and input magnitudes")
    return z


def features_from_records(hand, records, config=None):
    if not records:
        raise ValueError("cannot build features from an empty record batch")
    return build_features(
        hand,
        *[np.stack([getattr(record, name) for record in records]) for name in (
            "root_translation", "root_rotation", "q", "object_translation",
            "object_rotation", "object_center", "object_radius",
        )],
        config=config,
    )
