"""Export decoded world-frame poses without importing training or simulation.

Controls are supplied explicitly by the caller: decoding q does not define a
pregrasp or squeeze policy. Object scale belongs to the scene, never the hand.
"""
import hashlib
import json
from pathlib import Path


def export_request(decoded, output, *, protocol, hand_xml, joint_names, scenes,
                   controls, control_source, frame_transforms):
    """scenes maps scene_id to trusted scene path; controls maps (sample,candidate).

    Each controls entry contains pregrasp and squeeze, each with R/t/q in world
    metres/radians. joint_names describes q in all three poses. Failed decodes
    remain in the denominator as invalid_input rows.
    """
    if protocol not in ('floating', 'tabletop') or not control_source:
        raise ValueError('explicit protocol and control_source required')
    if len(joint_names) != 22 or len(set(joint_names)) != 22:
        raise ValueError('22 unique joint names required')
    source = Path(decoded).read_bytes()
    rows = [json.loads(x) for x in source.splitlines() if x.strip()]
    candidates, seen = [], set()
    for row in rows:
        key = row['sample_index'], row['candidate_index']
        if key in seen:
            raise ValueError('duplicate candidate identity')
        seen.add(key)
        item = dict(sample_index=key[0], candidate_index=key[1], scene=scenes[row['scene_id']])
        if row.get('ik_success') is not True or row.get('joint_limits_valid') is not True:
            item['invalid_reason'] = 'decode_or_joint_limits_failed'
        else:
            supplied = controls[key]
            item.update(grasp=world_hand_pose(row, **frame_transforms[row['scene_id']]),
                        pregrasp=supplied['pregrasp'], squeeze=supplied['squeeze'])
        candidates.append(item)
    request = dict(schema='dexgrasp-physics/v1', protocol=protocol, frame='world',
                   units=dict(length='m', angle='rad'), hand_xml=str(hand_xml),
                   hand_pose_frame='official_hand_root', frame_transforms=frame_transforms,
                   joint_names=list(joint_names), control_source=control_source,
                   decoded_sha256=hashlib.sha256(source).hexdigest(), candidates=candidates)
    with Path(output).open('x') as stream:
        json.dump(request, stream, allow_nan=False, indent=2)
    return request


def world_hand_pose(pose, *, world_from_input_R, world_from_input_t, official_from_decoded_R):
    """Undo hand-frame calibration and object-local coordinates exactly once.

    Input translations are already metric and object scale is already baked in.
    Controls supplied to export_request are separately defined in world/official
    hand coordinates, and are not transformed by this helper.
    """
    import numpy as np
    rotations = [np.asarray(x, float) for x in
                 (pose['R'], world_from_input_R, official_from_decoded_R)]
    for R in rotations:
        if (R.shape != (3, 3) or not np.isfinite(R).all()
                or not np.allclose(R.T@R, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(R), 1, atol=1e-5)):
            raise ValueError('frame transforms must be proper rotations')
    R, W, A = rotations
    t, origin = np.asarray(pose['t'], float), np.asarray(world_from_input_t, float)
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (t, origin)):
        raise ValueError('finite metric translations required')
    return dict(R=(W@R@A.T).tolist(), t=(W@t+origin).tolist(), q=pose['q'])
