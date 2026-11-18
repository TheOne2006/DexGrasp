"""Direction support, accuracy, and joint physical success for paired random commands."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from dexgrasp.test.frozen import decoded


def sphere_bins(directions):
    """32 equal-solid-angle bins, four in z and eight in azimuth."""
    z = np.clip(((directions[:, 2] + 1) * 2).astype(int), 0, 3)
    phi = np.arctan2(directions[:, 1], directions[:, 0]) % (2 * np.pi)
    return z * 8 + np.minimum(7, (phi * 4 / np.pi).astype(int))


def sphere_grid(size=8192):
    j = np.arange(size) + 0.5
    z = 1 - 2 * j / size
    radius = np.sqrt(1 - z * z)
    angle = j * (np.pi * (3 - np.sqrt(5)))
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))


def nearest_angles(labels, queries):
    tree = cKDTree(labels / np.linalg.norm(labels, axis=1, keepdims=True))
    chord, _ = tree.query(queries)
    return np.degrees(2 * np.arcsin(np.minimum(1, chord / 2)))


def analyze(root, inputs, force):
    plan = json.loads((root / "plan.json").read_text())
    data = Path(plan["root"])
    directions = np.load(data / "grasp_direction_cmd.npy", mmap_mode="r")
    test = directions[np.load(root / "test_rows.npy")]
    test_objects = np.load(root / "object_indices.npy")
    trial = [np.load(inputs / "A" / f"rank{r}" / "random/index.npz") for r in (0, 1)]
    objects = np.concatenate([s["objects"] for s in trial])
    order = np.argsort(np.concatenate([s["index"] for s in trial]))
    objects = objects[order]
    cmd = np.concatenate([s["directions"] for s in trial])[order]
    packed = np.concatenate([s["packed"] for s in trial])[order]
    for s in trial:
        s.close()
    nearest = np.empty(len(cmd))
    nearest_obs = np.empty(len(cmd))
    observations = np.load(data / "observation_index.npy", mmap_mode="r")
    test_obs = observations[np.load(root / "test_rows.npy")]
    trial_obs = observations[packed]
    grid = sphere_grid()
    thresholds = [5, 10, 20, 30]
    coverage = []
    for obj in np.unique(objects):
        mask = objects == obj
        ids = np.unique(trial_obs[mask])
        if len(ids) != 1:
            raise ValueError("Expected one frozen observation per object")
        record = {"object": plan["objects"][obj], "observation": int(ids[0])}
        queries = np.concatenate((cmd[mask], grid))
        for name, selection, distances in (
            ("object", test_objects == obj, nearest),
            ("observation", (test_objects == obj) & (test_obs == ids[0]), nearest_obs),
        ):
            angles = nearest_angles(test[selection], queries)
            distances[mask] = angles[: mask.sum()]
            record[name] = dict(
                labels=int(selection.sum()),
                area_fractions=[float((angles[mask.sum() :] <= t).mean()) for t in thresholds],
            )
        coverage.append(record)
    support = nearest <= 10
    bins = sphere_bins(cmd)
    train = directions[np.load(data / "train_rows.npy")]
    report = dict(
        train_bin_count=np.bincount(sphere_bins(train), minlength=32).tolist(),
        test_bin_count=np.bincount(sphere_bins(test), minlength=32).tolist(),
        random_bin_count=np.bincount(bins, minlength=32).tolist(),
        support_definition="nearest labeled test direction for the same object across its observations, <=10 deg",
        supported=int(support.sum()),
        total=len(support),
        nearest_angle_p50_p90_p95=np.quantile(nearest, [0.5, 0.9, 0.95]).tolist(),
        models={},
        sphere_coverage=dict(
            grid_size=len(grid),
            grid="Fibonacci sphere",
            thresholds_deg=thresholds,
            per_object=coverage,
            equal_object_mean={
                name: np.mean([r[name]["area_fractions"] for r in coverage], axis=0).tolist()
                for name in ("object", "observation")
            },
        ),
    )
    outcomes = np.load(force / "outcomes.npy")
    for i, g in enumerate("ABCD"):
        a = decoded(inputs, g, "random")
        passed = outcomes[i] == 1
        aligned = a["direction_pass"]
        joint = passed & aligned

        def stats(mask):
            return dict(
                n=int(mask.sum()),
                joint=int(joint[mask].sum()),
                joint_rate=float(joint[mask].mean()) if mask.any() else None,
            )

        by_object = np.bincount(objects, weights=joint) / np.bincount(objects)
        rng = np.random.default_rng(7301)
        ci = np.quantile(rng.choice(by_object, (4000, len(by_object))).mean(axis=1), [0.025, 0.975])
        report["models"][g] = dict(
            angular_mean_deg=float(a["angle_deg"].mean()),
            angular_p95_deg=float(np.quantile(a["angle_deg"], 0.95)),
            decoded=int(a["decoded"].sum()),
            fit=int(a["fit"].sum()),
            aligned=int(aligned.sum()),
            joint_ci95_bootstrap_objects=ci.tolist(),
            supported=stats(support),
            outside_support=stats(~support),
            same_observation_supported=stats(nearest_obs <= 10),
            same_observation_outside=stats(nearest_obs > 10),
            per_bin=[stats(bins == b) for b in range(32)],
        )
    (force / "coverage.json").write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "inputs", "force"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    analyze(args.root, args.inputs, args.force)
