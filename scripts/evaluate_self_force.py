"""Replay all frozen poses or a paired candidate-budget and controller ablation."""

from pathlib import Path

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dexgrasp.test.frozen import digest, save


def read(path):
    return json.loads(path.read_text())


def run(job):
    gid, ids, paired = job
    scene = engine.floating_scene(root / "assets", plan["groups"][gid])
    ctx = engine.make_scene(scene, hand, spec, settings)
    modes = ["derived", "no_squeeze", "hold"] if paired else ["derived"]
    a = np.zeros((len(poses), len(ids), len(modes)), dtype=np.uint8)
    sources = Counter()
    for j, (group, arr) in enumerate(poses.items()):
        for col, i in enumerate(ids):
            if arr[i]["decoded"]:
                results, source = engine.test_pose(ctx, arr[i], frame, settings, paired=paired)
                a[j, col] = [results[m] for m in modes]
                sources[group + " " + source] += 1
    return ids, a, sources


def main(mode, workers):
    paired = mode == "paired"
    out = root / mode
    out.mkdir(exist_ok=True)
    chunks = out / "chunks"
    chunks.mkdir(exist_ok=True)
    ids = np.load(root / "paired.npy") if paired else np.arange(len(objects))
    modes = ["derived", "no_squeeze", "hold"] if paired else ["derived"]
    protocol = dict(
        source=digest(Path(__file__)),
        engine_source=digest(Path(engine.__file__)),
        plan=digest(root / "plan.json"),
        hand=digest(hand),
        mode=mode,
        controls=modes,
        mujoco=mujoco.__version__,
        samples=len(ids),
        models=plan["models"],
    )
    receipt = out / "protocol.json"
    if receipt.exists() and read(receipt) != protocol:
        raise ValueError("resume protocol mismatch")
    save(receipt, protocol)
    jobs = []
    for gid in np.unique(geometry[ids]):
        group_ids = ids[geometry[ids] == gid]
        jobs.extend((gid, group_ids[i : i + 64], paired) for i in range(0, len(group_ids), 64))
    with ProcessPoolExecutor(workers) as pool:
        pending = [
            pool.submit(run, job) for job in jobs if not (chunks / f"{job[1][0]}.npz").exists()
        ]
        for j, f in enumerate(as_completed(pending), 1):
            indices, values, sources = f.result()
            path = chunks / f"{indices[0]}.npz"
            tmp = path.with_suffix(".partial")
            with tmp.open("wb") as stream:
                np.savez(stream, indices=indices, values=values, sources=json.dumps(sources))
            tmp.replace(path)
            if j % 50 == 0:
                print(mode, j, len(pending), flush=True)
    results = np.full((len(poses), len(ids), len(modes)), 255, np.uint8)
    for _, rows, _ in jobs:
        with np.load(chunks / f"{rows[0]}.npz") as z:
            np.testing.assert_array_equal(rows, z["indices"])
            results[:, np.searchsorted(ids, rows)] = z["values"]
    assert results.max() <= 1
    np.save(out / "outcomes.npy", results)
    summary = {}
    for j, g in enumerate(poses):
        summary[g] = {}
        for col, m in enumerate(modes):
            ok = results[j, :, col].astype(bool)
            summary[g][m] = dict(
                candidates=len(ok),
                successes=int(ok.sum()),
                objects=len(np.unique(objects[ids])),
                successful_objects=len(np.unique(objects[ids[ok]])),
                gsr=float(ok.mean()),
                osr=len(np.unique(objects[ids[ok]])) / len(np.unique(objects[ids])),
            )
    save(out / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("paired", "full"))
    for key in ("stages", "reference", "hand"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=80)
    parser.add_argument("--image-packages", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference.resolve()))
    if args.image_packages:
        sys.path.append(str(args.image_packages))
    from dexgrasp.test import dexonomy_control as engine
    import mujoco
    from omegaconf import OmegaConf
    from dexonomy.op.gen_grasp import GraspFilter

    root = args.stages
    plan = read(root / "plan.json")
    spec = read(Path(plan["hand_spec"]))
    settings = OmegaConf.load(args.reference / "dexonomy/config/op/grasp.yaml")
    hand = args.hand.resolve()
    frame = np.asarray(spec["official_from_grasp_rotation"])
    poses = {g: np.load(root / (g + ".npy"), mmap_mode="r") for g in plan["models"]}
    geometry = np.load(root / "groups.npy")
    objects = np.load(root / "objects.npy")
    main(args.mode, args.workers)
