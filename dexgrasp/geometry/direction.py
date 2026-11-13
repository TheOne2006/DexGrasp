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


def hand_direction(rotation, spec):
    """Scene-frame +Zh for v2; explicit legacy semantics for old datasets."""
    return np.asarray(rotation) @ np.asarray(direction_definition(spec)['local_axis'])
