"""Kinematic metrics, not collision or physical grasp success.

IK mean/max L2 and 5 mm fit threshold follow the definitions exposed by
reference/KPGrasp/ik.py. This implementation is independent. Diversity is
condition-local joint-angle sample standard deviation, unlike the pooled
physical-success pose statistic in TRO-Grasp.
"""
import time
import numpy as np

from ..geometry.decode import decode_keypoints


def reconstruction_metrics(generated, reconstructed, *, threshold_m=0.005):
    a, b = np.asarray(generated), np.asarray(reconstructed)
    if (a.shape != (21, 3) or b.shape != a.shape or
            not np.isfinite(a).all() or not np.isfinite(b).all() or threshold_m <= 0):
        raise ValueError('finite matching 21x3 keypoints and positive tolerance required')
    errors = np.linalg.norm(a-b, axis=-1)
    return dict(mean_keypoint_error_mm=float(errors.mean()*1000),
                max_keypoint_error_mm=float(errors.max()*1000),
                reconstruction_rmse_mm=float(np.sqrt(np.mean(errors**2))*1000),
                fit_success=bool(errors.max() <= threshold_m))


def joint_diversity(joints):
    q = np.asarray(joints, dtype=float)
    if len(q) < 2:
        return None
    if q.ndim != 2 or q.shape[1] != 22 or not np.isfinite(q).all():
        raise ValueError('finite N x 22 joint angles required')
    return float(q.std(axis=0, ddof=1).mean())


def decode_metrics(keypoints, hand, direction, *, threshold_m=0.005, max_nfev=100):
    started = time.perf_counter()
    row = dict(finite_output=bool(np.isfinite(keypoints).all()), decoded=False,
               fit_success=False, direction_pass=False, physical_success=None)
    try:
        decoded = decode_keypoints(keypoints, hand, direction, max_nfev=max_nfev)
        row.update(reconstruction_metrics(keypoints, decoded['reconstructed_keypoints'],
                                          threshold_m=threshold_m))
        row.update(decoded=True, q=decoded['q'].tolist(), R=decoded['R'].tolist(),
                   t=decoded['t'].tolist(), direction_error_deg=decoded['direction_error_deg'],
                   direction_pass=bool(decoded['direction_error_deg'] <= 10.),
                   optimizer_success=decoded['optimizer_success'],
                   joint_limits_valid=decoded['joint_limits_valid'],
                   legacy_rmse3mm_ik_success=decoded['ik_success'],
                   ik_nfev=decoded['ik_nfev'],
                   palm_registration_error_mm=1000*decoded['palm_registration_error_m'])
        row['fit_success'] = bool(row['fit_success'] and decoded['joint_limits_valid'])
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        row['error'] = str(exc)
    row['ik_seconds'] = time.perf_counter()-started
    return row


def summarize(rows, *, bootstrap_seed=7301, bootstrap_repeats=2000):
    """Equal condition weight within objects; equal object weight in main table.

    Failures remain in rate denominators; continuous errors have explicit
    coverage. Confidence intervals resample independent asset groups, retaining
    all objects in each group. These describe test-object sampling uncertainty,
    not variability across independently trained models.
    """
    if not rows:
        raise ValueError('no evaluation rows')
    grouped = {}
    for row in rows:
        grouped.setdefault(row['object_id'], {}).setdefault(row['condition_id'], []).append(row)
    per_object = []
    continuous = {'mean_keypoint_error_mm': 'mean_keypoint_error_mm',
                  'mean_max_keypoint_error_mm': 'max_keypoint_error_mm',
                  'direction_angular_error_deg': 'direction_error_deg'}
    for obj, conditions in sorted(grouped.items()):
        flat = [r for rs in conditions.values() for r in rs]
        out = dict(object_id=obj, asset_group=flat[0].get('asset_group', obj),
                   candidates=len(flat), conditions=len(conditions))
        for name, key in [('ik_fit_success_rate', 'fit_success'),
                          ('direction_pass_rate_10deg', 'direction_pass'),
                          ('decode_rate', 'decoded'), ('finite_output_rate', 'finite_output')]:
            out[name] = float(np.mean([np.mean([bool(r.get(key, False)) for r in rs])
                                       for rs in conditions.values()]))
        for name, key in continuous.items():
            means = [np.mean([r[key] for r in rs if r.get(key) is not None])
                     for rs in conditions.values() if any(r.get(key) is not None for r in rs)]
            out[name] = float(np.mean(means)) if means else None
            out[name+'_coverage'] = sum(r.get(key) is not None for r in flat)/len(flat)
        for name, passed_only in [('joint_angle_diversity_rad', False),
                                  ('ik_valid_joint_angle_diversity_rad', True)]:
            values = [joint_diversity([r['q'] for r in rs if r.get('decoded') and
                                      (not passed_only or r['fit_success'])])
                      for rs in conditions.values()]
            valid = [v for v in values if v is not None]
            out[name] = float(np.mean(valid)) if valid else None
            out[name+'_condition_coverage'] = len(valid)/len(values)
        per_object.append(out)
    macro = {}
    groups = sorted({r['asset_group'] for r in per_object})
    rng = np.random.default_rng(bootstrap_seed)
    sampled = rng.integers(len(groups), size=(bootstrap_repeats, len(groups)))
    for key in per_object[0]:
        if key in ('object_id', 'asset_group', 'candidates', 'conditions'):
            continue
        valid = [r[key] for r in per_object if r[key] is not None]
        sums = np.array([sum(r[key] for r in per_object if r['asset_group']==g and r[key] is not None)
                         for g in groups])
        counts = np.array([sum(r['asset_group']==g and r[key] is not None for r in per_object)
                           for g in groups])
        denominator = counts[sampled].sum(axis=1)
        estimates = sums[sampled].sum(axis=1)[denominator > 0]/denominator[denominator > 0]
        macro[key] = dict(mean=float(np.mean(valid)) if valid else None,
                          ci95=np.quantile(estimates, [.025, .975]).tolist() if len(estimates) else None,
                          objects_with_value=len(valid))
    distributions = {}
    for key in continuous.values():
        vals = [r[key] for r in rows if r.get(key) is not None]
        distributions[key] = dict(count=len(vals), total=len(rows),
                                  p50_p90_p95=np.quantile(vals, [.5, .9, .95]).tolist() if vals else None)
    return dict(candidates=len(rows), objects=len(per_object), macro=macro,
                candidate_distributions=distributions, per_object=per_object,
                physical_success_rate=None, collision_free_rate=None,
                note='Kinematic evaluation only. Undefined diversity is null, not zero; failures remain in rate denominators.')
