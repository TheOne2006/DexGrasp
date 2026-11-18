"""Versioned direction semantics, independent of palm gap measurement."""
import numpy as np

WRIST_Z = 'official_wrist_positive_z_v2'
LEGACY = 'legacy_palm_normal_v1'


def direction_definition(spec):
    definition = spec.get('direction_condition')
    if definition is None:
        return dict(kind=LEGACY, local_axis=spec['approach_axis'])
    if definition.get('kind') != WRIST_Z:
        raise ValueError('Unknown direction condition version')
    axis = np.asarray(definition.get('local_axis'), dtype=float)
    transform = np.asarray(spec['official_from_grasp_rotation'], dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or not np.allclose(axis, transform.T @ [0., 0., 1.], atol=1e-8):
        raise ValueError('Direction must be official +Zh expressed in the grasp frame')
    if not np.isclose(np.linalg.norm(axis), 1., atol=1e-8):
        raise ValueError('Direction axis must be unit length')
    return definition


def official_wrist_definition(spec):
    """Recover official +Zh from the hand calibration, including older formal hand specs."""
    basis = np.asarray(spec['official_from_grasp_rotation'], dtype=float)
    if (basis.shape != (3, 3) or not np.allclose(basis.T @ basis, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(basis), 1., atol=1e-6)):
        raise ValueError('official_from_grasp_rotation must be a proper rotation')
    if spec.get('direction_condition') is not None:
        return direction_definition(spec)
    return dict(kind=WRIST_Z, local_axis=basis[2].tolist())


def hand_direction(rotation, spec):
    """Scene-frame +Zh for v2; explicit legacy semantics for old datasets."""
    return np.asarray(rotation) @ np.asarray(direction_definition(spec)['local_axis'])


def raw_anchor_direction_error(keypoints, hand, command):
    """Estimate the wrist axis directly from anchor offsets before rigid registration.

    Express the calibrated axis as a linear combination of centered template anchors.
    The same weights recover R @ axis exactly on rigid data, without fitting R to a draw.
    """
    template = hand.palm_template - hand.palm_template.mean(0)
    axis = np.asarray(direction_definition(hand.spec)['local_axis'])
    weights = np.linalg.lstsq(template.T, axis, rcond=None)[0]
    if not np.allclose(template.T @ weights, axis, atol=1e-6):
        return None
    anchors = np.asarray(keypoints)[hand.palm_indices]
    direction = weights @ (anchors - anchors.mean(0))
    norm = np.linalg.norm(direction) * np.linalg.norm(command)
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    return float(np.degrees(np.arccos(np.clip((direction @ command) / norm, -1., 1.))))
