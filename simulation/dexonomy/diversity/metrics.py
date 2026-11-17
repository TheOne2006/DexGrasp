"""CPU streaming means over all global nonself pairs of weighted descriptors.

Exact mode enumerates every unordered pair once. Sample mode draws ordered
nonself pairs uniformly with replacement; both directions share the kernel.
No object/group weighting, diagonal term, reciprocal, or RKE is involved.
"""
from __future__ import annotations

import math
from numbers import Integral
import numpy as np

BLOCKS = ("joint", "translation", "rotation", "hand")


def _integer(value, name, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _options(mode, pairs_per_grasp, num_pairs, seed, batch_size):
    if mode not in ("exact", "sample"):
        raise ValueError("mode must be exact or sample")
    return (_integer(pairs_per_grasp, "pairs_per_grasp"),
            None if num_pairs is None else _integer(num_pairs, "num_pairs"),
            _integer(seed, "seed", minimum=0), _integer(batch_size, "batch_size"))


def _sample_pairs(n, count, batch_size, seed):
    # One integer identifies one ordered pair, preserving the random stream
    # across batch sizes. The budget is never clipped to the population size.
    rng = np.random.default_rng(seed)
    for start in range(0, count, batch_size):
        ids = rng.integers(n * (n - 1), size=min(batch_size, count - start))
        first = ids // (n - 1)
        second = ids % (n - 1)
        second += second >= first
        yield first, second


def _exact_pairs(n, batch_size):
    for first in range(n - 1):
        for start in range(first + 1, n, batch_size):
            second = np.arange(start, min(start + batch_size, n), dtype=np.int64)
            yield np.full(second.shape, first, dtype=np.int64), second


def _squared_kernels(features, first, second):
    # Direct differences avoid cancellation from subtracting large Gram terms.
    # Overflow for finite, extremely distant features gives a valid zero kernel.
    with np.errstate(over="ignore", under="ignore"):
        difference = features[first] - features[second]
        translation = np.einsum("ij,ij->i", difference[:, :3], difference[:, :3])
        rotation = np.einsum("ij,ij->i", difference[:, 3:12], difference[:, 3:12])
        hand = np.einsum("ij,ij->i", difference[:, 12:], difference[:, 12:])
        distances = np.column_stack((translation + rotation + hand, translation, rotation, hand))
        return np.exp(-distances)


def evaluate_pair_features(features, *, mode="exact", pairs_per_grasp=100,
                           num_pairs=None, seed=0, batch_size=16384, diagnostics=True):
    """Evaluate Q=1-mean(exp(-squared_distance)) over the whole feature array.

    Features already include physical scales and square-root block weights.
    They have shape (N,12+3*L), default L=21. Columns are translation3,
    rotation9 and hand3L. This API applies no additional scaling or grouping.
    Only global N<2 is unavailable. Sample B is num_pairs or c*N, including
    records whose object/condition happens to contain only one grasp.
    """
    pairs_per_grasp, num_pairs, seed, batch_size = _options(mode, pairs_per_grasp, num_pairs, seed, batch_size)
    if not isinstance(diagnostics, bool):
        raise ValueError("diagnostics must be boolean")
    try:
        z = np.asarray(features, dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise ValueError("features must be a finite numeric 2D array") from error
    if z.ndim != 2 or z.shape[1] < 15 or (z.shape[1] - 12) % 3:
        raise ValueError("features must have shape (N,12+3*L), L>=1")
    if not np.isfinite(z).all():
        raise ValueError("features must be finite")
    n = len(z)
    u = n * (n - 1) // 2
    if n < 2:
        count = 0
    elif mode == "exact":
        count = u
    elif num_pairs is not None:
        count = num_pairs
    else:
        count = pairs_per_grasp * n
    mean = np.zeros(4, dtype=np.float64)
    centered_sum_squares = np.zeros(4, dtype=np.float64)
    seen = 0
    if n >= 2:
        batches = _exact_pairs(n, batch_size) if mode == "exact" else _sample_pairs(n, count, batch_size, seed)
        for first, second in batches:
            values = _squared_kernels(z, first, second)
            batch_count = len(values)
            batch_mean = values.mean(axis=0)
            centered = values - batch_mean
            batch_ss = np.einsum("ij,ij->j", centered, centered)
            total = seen + batch_count
            delta = batch_mean - mean
            centered_sum_squares += batch_ss + delta * delta * (seen * batch_count / total)
            mean += delta * (batch_count / total)
            seen = total
    if seen != count:
        raise RuntimeError("internal global pair enumeration count mismatch")
    mean = np.clip(mean, 0., 1.)
    epsilon = math.sqrt(math.log(40.) / (2 * count)) if n >= 2 and mode == "sample" else 0.
    scores = {}
    for index, block in enumerate(BLOCKS):
        if not diagnostics and block != "joint":
            continue
        mu = float(mean[index]) if n >= 2 else None
        lo, hi = (max(0., mu-epsilon), min(1., mu+epsilon)) if mu is not None else (None, None)
        if n < 2:
            standard_error = None
        elif mode == "exact":
            standard_error = 0.0
        elif count > 1:
            variance_sum = max(0.0, float(centered_sum_squares[index]))
            standard_error = math.sqrt(variance_sum / (count - 1) / count)
        else:
            standard_error = None
        scores[block] = {"mu": mu, "Q": 1-mu if mu is not None else None,
            "pair_mc_standard_error": standard_error,
            "mu_mc_95_interval": [lo,hi] if mu is not None else None,
            "Q_mc_95_interval": [1-hi,1-lo] if mu is not None else None}
    if n < 2:
        sampling = "none"
        uncertainty_method = "undefined"
    elif mode == "sample":
        sampling = "uniform_ordered_nonself_with_replacement"
        uncertainty_method = "hoeffding"
    else:
        sampling = "all_unordered_nonself_pairs_once"
        uncertainty_method = "exact"
    return {"schema": "global-pair-mean-diversity/v1", "pair_scope": "global",
        "status": "complete" if n >= 2 else "no_nonself_pairs",
        "definition": "mu=mean_{all global i<j} exp(-||z_i-z_j||^2); Q=1-mu",
        "aggregation": "uniform global nonself record pairs; no object or condition averaging",
        "dataset_diversity": scores["joint"]["Q"], "dataset_collision_mean": scores["joint"]["mu"],
        "total_N": n, "total_U": u, "total_B": count, "n": n, "num_pairs": count,
        "num_possible_unordered_pairs": u, "num_possible_ordered_pairs": n*(n-1),
        "feature_dimension": z.shape[1], "requested_mode": mode,
        "mode": mode if n >= 2 else "undefined", "seed": seed, "batch_size": batch_size,
        "requested_num_pairs": num_pairs, "pairs_per_grasp": pairs_per_grasp,
        "pair_budget_scope": "one total budget for the whole pool; fallback c*N uses ALL selected records",
        "pair_sampling": sampling,
        "uncertainty": {"method": uncertainty_method,
            "scope": "pair-draw Monte Carlo conditional on the fixed entire collection; not subset, object or population uncertainty",
            "confidence": .95 if mode == "sample" and n >= 2 else None,
            "coverage": "marginal_per_reported_score",
            "pair_mc_standard_error": "SE of sampled global squared-kernel mean and Q; null for B=1 or N<2"},
        "scores": scores}
