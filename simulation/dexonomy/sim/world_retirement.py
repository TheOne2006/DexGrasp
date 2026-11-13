"""Checkpoint lifecycle independent of the physics backend and task protocol.

A terminal direction is not a complete trajectory. Unobserved poses remain NaN
in the binary audit; JSON consumers receive explicit checkpoint statuses.
"""
import numpy as np
from dexonomy.util.np_util import np_get_delta_pose

SCHEMA = 'force_closure_6d_retirement_v1'


def inspect_checkpoint(target, pose, translation_threshold, rotation_threshold):
    if not np.isfinite(target).all() or not np.isfinite(pose).all():
        raise ValueError('non-finite active checkpoint')
    trans, rot = np_get_delta_pose(target, pose)
    if not np.isfinite([trans, rot]).all():
        raise ValueError('non-finite active checkpoint metric')
    return {'translation_error_m': float(trans), 'rotation_error_deg': float(rot),
            'passes': bool(trans < translation_threshold and rot < rotation_threshold)}


def direction_result(target, poses, seen, translation_threshold, rotation_threshold):
    mask = np.asarray(seen)
    if np.asarray(poses).shape != (10, 7) or mask.shape != (10,) or not np.isin(mask, [0, 1]).all():
        raise ValueError('ten explicit checkpoint slots required')
    seen = mask.astype(bool)
    count = int(seen.sum())
    if not 1 <= count <= len(seen) or not np.all(seen[:count]) or np.any(seen[count:]):
        raise ValueError('checkpoint observations must be a nonempty prefix')
    values = [inspect_checkpoint(target, p, translation_threshold, rotation_threshold)
              for p in poses[:count]]
    if any(not x['passes'] for x in values[:-1]):
        raise ValueError('trajectory advanced beyond a failed checkpoint')
    passed = values[-1]['passes']
    if passed and count != len(seen):
        raise ValueError('incomplete passing direction')
    return {'status': 'COMPLETED' if passed else 'TERMINATED_PHYSICAL_FAILURE',
            'passes': passed, 'completed_holds': count,
            'checkpoint_status': ['OBSERVED'] * count + ['NOT_RUN'] * (len(seen) - count),
            'checkpoints': values,
            'terminal_hold': count, 'terminal_reason': None if passed else 'pose_threshold',
            'max_translation_error_m': max(x['translation_error_m'] for x in values),
            'max_rotation_error_deg': max(x['rotation_error_deg'] for x in values)}


def candidate_metrics(target, poses, seen, directions, trans, rot):
    if np.asarray(poses).shape != (6, 10, 7) or np.asarray(seen).shape != (6, 10) or np.asarray(directions).shape != (6, 3):
        raise ValueError('six complete direction mappings required')
    rows = [dict(direction_result(target, p, s, trans, rot), direction=d.astype(float).tolist())
            for p, s, d in zip(poses, seen, directions)]
    if len(rows) != 6:
        raise ValueError('six explicit direction records required')
    wt = max(x['max_translation_error_m'] for x in rows)
    wr = max(x['max_rotation_error_deg'] for x in rows)
    return dict(schema=SCHEMA, type='force_closure_6d', passes=all(x['passes'] for x in rows),
                directions_completed=sum(x['status'] == 'COMPLETED' for x in rows),
                directions_evaluated=6, directions_terminal=6, directions_required=6,
                direction_results=rows, worst_translation_error_m=wt,
                worst_rotation_error_deg=wr, translation_threshold_m=trans,
                rotation_threshold_deg=rot, stability_score=max(wt/trans, wr/rot))
