"""Score native physical replays for each frozen set of direction and noise draws."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
from dexgrasp.test.frozen import digest, save
from scripts.analyze_direction_coverage import analyze as analyze_coverage
from scripts.analyze_noise_diversity import analyze as analyze_shape


def summarize(source, root):
    plan = json.loads((source / "plan.json").read_text())
    if digest(source / "objects.npy") != plan["object_map"]:
        raise ValueError("The frozen object mapping changed")
    objects = np.load(source / "objects.npy")
    groups = plan["groups"]
    truth = np.full((len(groups), len(objects)), 255, dtype=np.uint8)
    for sid, rows in plan["sources"].items():
        path = source / "verdicts" / f"{sid}.json"
        records = json.loads(path.read_text())
        if len(records) != len(rows) * len(groups):
            raise ValueError(f"Unexpected candidate count in {path}")
        for row in rows:
            i = row["index"]
            if row["object"] != objects[i] or (truth[:, i] != 255).any():
                raise ValueError(f"Incorrect candidate mapping at row {i}")
            for j, group in enumerate(groups):
                value = records[f"{group}/{i}"]
                if not isinstance(value, bool):
                    raise TypeError(f"Expected a physical verdict at row {i}")
                truth[j, i] = value
    if (truth > 1).any():
        raise ValueError("Missing candidate verdicts")
    poses = {g: np.load(source / f"{g}.npy", mmap_mode="r") for g in groups}
    all_sets = {}
    for name, subset in plan["sets"].items():
        sel = slice(subset["start"], subset["stop"])
        out = source / name
        out.mkdir(exist_ok=True)
        trials = truth[:, sel].astype(bool)
        np.save(out / "outcomes.npy", trials)
        ids = objects[sel]
        total_objects = len(np.unique(ids))
        report = {}
        for group, passed in zip(groups, trials, strict=True):
            aligned = poses[group]["direction_pass"][sel]
            joint = passed & aligned
            wins = len(np.unique(ids[passed]))
            report[group] = dict(n=len(passed), successes=int(passed.sum()), gsr=float(passed.mean()),
                                 objects=total_objects, successful_objects=wins, osr=wins / total_objects,
                                 aligned=int(aligned.sum()), joint=int(joint.sum()),
                                 aligned_and_success=float(joint.mean()))
        save(out / "summary.json", report)
        inputs = Path(subset["inputs"])
        if digest(inputs / "design.json") != subset["design"]:
            raise ValueError(f"The inference design changed for {name}")
        if subset["mode"] == "random":
            analyze_coverage(root, inputs, out)
        else:
            analyze_shape(root, inputs, out)
        all_sets[name] = report
    save(source / "conditioned_summary.json", dict(sets=all_sets, groups=groups,
         n_per_model=len(objects), source_plan=digest(source / "plan.json"),
         analysis=digest(__file__)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    summarize(args.input, args.root)
