"""Replay existing random-direction and fixed-direction noise draws with three-stage controls."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dexgrasp.test.frozen import digest, save
from dexgrasp.test.force_summary import solved_force_metrics


def read(path):
    return json.loads(path.read_text())


def evaluate_group(job):
    gid, indices = job
    scene = engine.floating_scene(stages / "assets", plan["groups"][gid])
    context = engine.make_scene(scene, hand_xml, spec, settings)
    result = np.zeros((4, len(indices)), dtype=bool)
    solved = np.zeros_like(result)
    for j, array in enumerate(poses.values()):
        for col, i in enumerate(indices):
            if array[i]["decoded"]:
                outcomes, method = engine.test_pose(context, array[i], rotation, settings)
                result[j, col] = outcomes["derived"]
                solved[j, col] = method == "contact_qp"
    return indices, result, solved


def main(args):
    global plan, poses, rotation, hand_xml, settings, stages, engine, spec
    sys.path.insert(0, str(args.reference.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if args.image_packages:
        sys.path.append(str(args.image_packages))
    from dexgrasp.test.frozen import decoded
    from dexgrasp.test import dexonomy_control as engine
    import mujoco
    from dexonomy.op.gen_grasp import GraspFilter
    from omegaconf import OmegaConf

    stages = args.stages.resolve()
    plan = read(stages / "plan.json")
    parent = read(args.root / "plan.json")
    settings = OmegaConf.load(args.reference / "dexonomy/config/op/grasp.yaml")
    hand_xml = args.hand.resolve()
    spec = read(Path(plan["hand_spec"]))
    rotation = np.asarray(spec["official_from_grasp_rotation"])
    poses = {g: decoded(args.inputs, g, args.mode) for g in "ABCD"}
    n = len(poses["A"])
    packed, objects = (np.empty(n, dtype=int) for _ in range(2))
    for rank in (0, 1):
        with np.load(args.inputs / "A" / f"rank{rank}" / args.mode / "index.npz") as shard:
            packed[rank::2], objects[rank::2] = shard["packed"], shard["objects"]
    obs = np.load(Path(parent["root"]) / "observation_index.npy", mmap_mode="r")[packed]
    centers = np.load(args.root / "quality/centers.npy")[obs]
    for array in poses.values():
        array["t"] += centers
    by_obs = {o: g["id"] for g in plan["groups"] for o in g["observations"]}
    geometry = np.asarray([by_obs[o] for o in obs])
    out = args.output
    chunks = out / "chunks"
    chunks.mkdir(exist_ok=True, parents=True)
    protocol = dict(
        mode=args.mode,
        stages=digest(stages / "plan.json"),
        engine=digest(engine.__file__),
        source=digest(__file__),
        inputs=digest(args.inputs / "design.json"),
        hand=digest(hand_xml),
        count=n,
        objects=int(len(np.unique(objects))),
        mujoco=mujoco.__version__,
    )
    if (out / "protocol.json").exists() and read(out / "protocol.json") != protocol:
        raise ValueError("Resume configuration changed")
    save(out / "protocol.json", protocol)
    jobs = [(g, np.flatnonzero(geometry == g)) for g in np.unique(geometry)]
    with ProcessPoolExecutor(args.workers) as pool:
        pending = [
            pool.submit(evaluate_group, job)
            for job in jobs
            if not (chunks / f"{job[0]}.npz").exists()
        ]
        for future in as_completed(pending):
            ids, values, solved = future.result()
            path = chunks / f"{geometry[ids[0]]}.npz"
            tmp = path.with_suffix(".partial")
            with tmp.open("wb") as f:
                np.savez(f, ids=ids, values=values, solved=solved)
            tmp.replace(path)
    outcomes = np.zeros((4, n), dtype=bool)
    solved = np.zeros_like(outcomes)
    for gid, ids in jobs:
        with np.load(chunks / f"{gid}.npz") as z:
            np.testing.assert_array_equal(ids, z["ids"])
            outcomes[:, ids] = z["values"]
            solved[:, ids] = z["solved"]
    np.save(out / "outcomes.npy", outcomes)
    np.save(out / "both_solved.npy", solved)
    save(out / "conditional_summary.json", {
        group: solved_force_metrics(solved[j], outcomes[j], objects) for j, group in enumerate(poses)
    })
    report = {}
    for j, name in enumerate(poses):
        passed = outcomes[j]
        aligned = poses[name]["direction_pass"]
        report[name] = dict(
            n=n,
            successes=int(passed.sum()),
            gsr=float(passed.mean()),
            osr=len(np.unique(objects[passed])) / len(np.unique(objects)),
            aligned_and_success=float((passed & aligned).mean()),
        )
    save(out / "summary.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "inputs", "stages", "reference", "output", "hand"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--image-packages", type=Path)
    parser.add_argument("--mode", choices=["random", "noise"], required=True)
    parser.add_argument("--workers", type=int, default=32)
    main(parser.parse_args())
