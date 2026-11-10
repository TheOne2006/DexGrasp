"""Proper rigid registration followed by bounded finger IK; no physical-success claim."""
import numpy as np


def rigid_registration(template, observed):
    template, observed = np.asarray(template, float), np.asarray(observed, float)
    if template.shape != observed.shape or template.ndim != 2 or template.shape[1] != 3 or len(template) < 3:
        raise ValueError('matching Nx3 palm anchors required')
    if not np.isfinite(template).all() or not np.isfinite(observed).all():
        raise ValueError('nonfinite palm anchors')
    a, b = template-template.mean(0), observed-observed.mean(0)
    if min(np.linalg.matrix_rank(a, tol=1e-8), np.linalg.matrix_rank(b, tol=1e-8)) < 2:
        raise ValueError('degenerate palm anchors')
    u, _, vh = np.linalg.svd(a.T @ b)
    correction = np.diag([1., 1., np.linalg.det(vh.T @ u.T)])
    rotation = vh.T @ correction @ u.T
    translation = observed.mean(0) - rotation @ template.mean(0)
    error = np.sqrt(np.mean(np.sum((template @ rotation.T+translation-observed)**2, -1)))
    return rotation, translation, float(error)


def decode_keypoints(keypoints_m, hand, direction, *, q_initial=None, tolerance_m=0.003, max_nfev=100):
    from scipy.optimize import least_squares
    k = np.asarray(keypoints_m, float)
    if k.shape != (21, 3) or not np.isfinite(k).all():
        raise ValueError('21 finite keypoints in metres required')
    if tolerance_m <= 0 or max_nfev < 1:
        raise ValueError('positive IK tolerance and iteration budget required')
    rotation, translation, palm_error = rigid_registration(hand.palm_template, k[hand.palm_indices])
    target_local = (k-translation) @ rotation
    fingers = np.ones(21, dtype=bool)
    fingers[hand.palm_indices] = False
    initial = (hand.lower+hand.upper)/2 if q_initial is None else np.asarray(q_initial, float)
    if initial.shape != (22,) or not np.isfinite(initial).all() or (initial < hand.lower).any() or (initial > hand.upper).any():
        raise ValueError('IK initial joint angles outside limits')
    result = least_squares(lambda q: (hand.fk(q)[fingers]-target_local[fingers]).ravel(), initial,
                           jac=lambda q: hand.fk(q, jacobian=True)[1][fingers].reshape(-1, 22),
                           bounds=(hand.lower, hand.upper), max_nfev=max_nfev,
                           ftol=1e-9, xtol=1e-9, gtol=1e-9)
    rebuilt = hand.world_keypoints(rotation, translation, result.x)
    error = float(np.sqrt(np.mean(np.sum((rebuilt-k)**2, -1))))
    d = np.asarray(direction, float)
    if d.shape != (3,) or not np.isfinite(d).all() or np.linalg.norm(d) < 1e-8:
        raise ValueError('nonzero direction required')
    d /= np.linalg.norm(d)
    angle = float(np.degrees(np.arccos(np.clip(rotation[:, 2] @ d, -1, 1))))
    return dict(R=rotation, t=translation, q=result.x, reconstructed_keypoints=rebuilt,
                ik_success=bool(result.success and error <= tolerance_m and palm_error <= tolerance_m),
                optimizer_success=bool(result.success), ik_nfev=int(result.nfev),
                reconstruction_error_m=error, palm_registration_error_m=palm_error,
                direction_error_deg=angle, joint_limits_valid=bool((result.x >= hand.lower).all() and (result.x <= hand.upper).all()),
                physical_success=None, collision_free=None)
