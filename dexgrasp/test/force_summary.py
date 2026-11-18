"""Aggregate both published force tests with explicit object and scale-specific budgets."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np


def summarize_official(inputs, output, io_workers=24):
    plan = json.loads((Path(inputs) / "plan.json").read_text())
    scenes = plan["scenes"]
    folder = Path(output) / "scenes"
    groups = "ABCD"
    backends = (("dexonomy", 0), ("dexgraspbench", 2))
    ntry = plan["candidates_per_scene"]
    ok = np.empty((len(scenes), len(groups), 2, ntry), dtype=bool)
    controls = []

    def read(i):
        return json.loads((folder / f"{i}.json").read_text())

    with ThreadPoolExecutor(io_workers) as pool:
        for i, data in enumerate(pool.map(read, range(len(scenes)))):
            scene_controls = []
            for j, name in enumerate(groups):
                values = data[name]
                if len(values) != ntry or any(
                    type(v[col]) is not bool for v in values for _, col in backends
                ):
                    raise ValueError(f"Incomplete scene {i}, model {name}")
                for b, (_, col) in enumerate(backends):
                    ok[i, j, b] = [v[col] for v in values]
                scene_controls.append(Counter(v[1] for v in values))
            controls.append(scene_controls)
    key = [(s["source"], s["object"]) for s in scenes]
    unique = {k: i for i, k in enumerate(sorted(set(key)))}
    objects = np.asarray([unique[k] for k in key])
    is_seen = np.asarray([s["training_overlap"] for s in scenes])
    datasets = np.asarray([s["source"] for s in scenes])
    selectors = {"all": np.ones(len(scenes), bool), "unseen": ~is_seen, "training_overlap": is_seen}
    selectors.update({str(k): datasets == k for k in sorted(set(datasets))})
    overlap_path = Path(inputs) / "mesh_overlap.json"
    if overlap_path.is_file():
        inventory = json.loads(overlap_path.read_text())["objects"]
        matches = [inventory[source][obj] for source, obj in key]
        if not np.array_equal([entry["id_overlap"] for entry in matches], is_seen):
            raise ValueError("Mesh inventory and evaluation plan disagree about training IDs")
        selectors["unseen_id_and_mesh"] = ~is_seen & ~np.asarray(
            [entry["sha_overlap"] for entry in matches], dtype=bool
        )
    summary = {}
    for j, group in enumerate(groups):
        summary[group] = {}
        for b, (name, _) in enumerate(backends):
            sections = {}
            for label, mask in selectors.items():
                v = ok[mask, j, b]
                success = v.any(axis=1)
                obj = objects[mask]
                n = len(np.unique(obj))
                aggregate = Counter()
                for i in np.flatnonzero(mask):
                    aggregate.update(controls[i][j])
                sections[label] = dict(
                    total=v.size,
                    objects=n,
                    successes=int(v.sum()),
                    successful_objects=len(np.unique(obj[success])),
                    gsr=float(v.mean()) if v.size else None,
                    osr=len(np.unique(obj[success])) / n if n else None,
                    control_sources=dict(aggregate),
                    scenes=len(v),
                    successful_scenes=int(success.sum()),
                    scene_success_rate=float(success.mean()) if len(v) else None,
                )
            summary[group][name] = sections
    return summary
