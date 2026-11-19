"""Conditional six-force metrics for the frozen four-model evaluation."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dexgrasp.test.force_summary import solved_force_metrics


def read(path):
    return json.loads(path.read_text())


def paired(root):
    folder = root / "source_tasks_paired"
    plan = read(folder / "plan.json")
    observations = {g: [] for g in plan["groups"]}
    for sid, rows in plan["sources"].items():
        controls = read(folder / "controller" / f"{sid}.json")
        verdicts = read(folder / "verdicts" / f"{sid}.json")
        for g, cases in observations.items():
            for row in rows:
                key = f"{g}/{row['index']}"
                value = verdicts[key]
                if not isinstance(value, bool):
                    raise ValueError(f"Missing verdict at {key}")
                cases.append((controls.get(key) == "resident_qp", value, row["object"]))
    results = {g: solved_force_metrics(*zip(*cases)) for g, cases in observations.items()}
    audit = read(folder / "controller_audit.json")["class_counts"]
    for g, rec in results.items():
        qp = audit[g]["all"]["resident_qp"]
        assert (rec["solved"], rec["passed"]) == (qp["attempts"], qp["successes"])
    return results


def official(root):
    folder = root / "official"
    plan = read(folder / "plan.json")
    scenes = plan["scenes"]
    budget = plan["candidates_per_scene"]
    group_keys = [(r["source"], r["object"]) for r in scenes]
    obj_lookup = {k: i for i, k in enumerate(sorted(set(group_keys)))}
    obj_ids = np.repeat([obj_lookup[k] for k in group_keys], budget)
    scene_ids = np.repeat(np.arange(len(scenes)), budget)
    selected = np.empty((4, len(scenes), budget), dtype=bool)
    passed = np.empty_like(selected)

    def load_scene(i):
        return read(folder / "physics/scenes" / f"{i}.json")

    with ThreadPoolExecutor(24) as pool:
        for i, scene in enumerate(pool.map(load_scene, range(len(scenes)))):
            for j, g in enumerate("ABCD"):
                rows = scene[g]
                if len(rows) != budget or any(type(v[2]) is not bool for v in rows):
                    raise ValueError(f"Incomplete benchmark data for {i}/{g}")
                selected[j, i] = [r[1] == "contact_qp" for r in rows]
                passed[j, i] = [r[2] for r in rows]
    audit = read(folder / "physics/controller_outcomes.json")["models"]
    results = {}
    for j, g in enumerate("ABCD"):
        valid = selected[j].ravel()
        ok = passed[j].ravel()
        results[g] = solved_force_metrics(valid, ok, scene_ids)
        results[g]["object_at_60"] = solved_force_metrics(valid, ok, obj_ids)
        rec = audit[g]["contact_qp"]
        assert (results[g]["solved"], results[g]["passed"]) == (rec["n"], rec["bench_success"])
    return results


def random_directions(root):
    inputs = root / "random_inputs"
    arrays = {}
    for name in "ABCD":
        with np.load(inputs / name / "rank0/random/index.npz") as a, np.load(
            inputs / name / "rank1/random/index.npz"
        ) as b:
            order = np.argsort(np.r_[a["index"], b["index"]])
            for k in a.files:
                values = np.concatenate([a[k], b[k]])[order]
                if name == "A":
                    arrays[k] = values
                else:
                    np.testing.assert_array_equal(arrays[k], values)
    ids, units, dirs = (arrays[k] for k in ("index", "objects", "directions"))
    np.testing.assert_array_equal(ids, np.arange(len(ids)))
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1, atol=1e-6)
    bins = np.clip(((dirs[:, 2] + 1) * 2).astype(int), 0, 3) * 8 + np.minimum(
        7, ((np.arctan2(dirs[:, 1], dirs[:, 0]) % (2 * np.pi)) * 4 / np.pi).astype(int)
    )
    trials = root / "random_stages"
    solved, passed = [np.load(trials / p) for p in ("both_solved.npy", "outcomes.npy")]
    np.testing.assert_array_equal(
        np.bincount(bins, minlength=32), read(trials / "coverage.json")["random_bin_count"]
    )
    np.testing.assert_array_equal(
        passed.sum(axis=1), [read(trials / "summary.json")[g]["successes"] for g in "ABCD"]
    )
    assert solved.shape == passed.shape == (4, len(ids))
    return {g: solved_force_metrics(solved[j], passed[j], units) for j, g in enumerate("ABCD")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("section", choices=("paired", "official", "random"))
    args = parser.parse_args()
    analysts = {"paired": paired, "official": official, "random": random_directions}
    result = analysts[args.section](args.evaluation)
    out = args.evaluation / "conditional"
    out.mkdir(exist_ok=True)
    path = out / f"{args.section}.json"
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)
