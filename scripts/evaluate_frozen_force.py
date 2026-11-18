"""Replay a complete frozen test or direction trial with native MuJoCo."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import subprocess

import mujoco
import numpy as np

from dexgrasp.geometry.hand import URDFHand
from dexgrasp.test.frozen import decoded, digest, save
from dexgrasp.test.physics import PoseEvaluator, PROTOCOL


def read(path):
    return json.loads(path.read_text())


def simulate(job):
    group, ids = job
    evaluator = PoseEvaluator(hand, quality["hand_xml"], group["pieces"], group["scale"])
    outcomes = np.zeros((len(models), len(ids)), dtype=np.uint8)
    for row, poses in enumerate(models.values()):
        for col, index in enumerate(ids):
            p = poses[index]
            if p["decoded"]:
                result = evaluator.evaluate(p["R"], p["t"] + centers[observations[index]], p["q"])
                outcomes[row, col] = 1 if result["single_pose_force_success"] else 2
    return ids, outcomes


def main(root, inputs, output, mode, workers):
    global hand, quality, models, centers, observations
    plan = read(root / "plan.json")
    quality = read(root / "quality/protocol.json")
    if quality["parent_plan_sha256"] != digest(root / "plan.json"):
        raise ValueError("quality geometry is from a different plan")
    data = Path(plan["root"])
    hand = URDFHand(quality["hand_spec"])
    models = {g: decoded(inputs, g, mode) for g in "ABCD"}
    with (
        np.load(inputs / "A/rank0" / mode / "index.npz") as a,
        np.load(inputs / "A/rank1" / mode / "index.npz") as b,
    ):
        n = len(a["index"]) + len(b["index"])
        packed, objects = (np.empty(n, dtype=int) for _ in range(2))
        for rank, shard in enumerate([a, b]):
            packed[rank::2], objects[rank::2] = shard["packed"], shard["objects"]
    if mode == "test":
        np.testing.assert_array_equal(packed, np.load(root / "test_rows.npy"))
        np.testing.assert_array_equal(objects, np.load(root / "object_indices.npy"))
    observations = np.load(data / "observation_index.npy", mmap_mode="r")[packed]
    centers = np.load(root / "quality/centers.npy")
    groups = {o: g["id"] for g in quality["groups"] for o in g["observations"]}
    labels = np.array([groups[o] for o in observations])
    jobs = []
    for group in quality["groups"]:
        ids = np.flatnonzero(labels == group["id"])
        jobs.extend((group, ids[i : i + 96]) for i in range(0, len(ids), 96))
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        physics=PROTOCOL,
        mujoco=mujoco.__version__,
        split="test",
        mode=mode,
        plan_sha256=digest(root / "plan.json"),
        quality_sha256=digest(root / "quality/protocol.json"),
        decoded={
            g: [digest(inputs / g / f"rank{r}" / mode / "decoded.npy") for r in (0, 1)]
            for g in models
        },
        code=digest(__file__),
        physics_code=digest(Path(__file__).resolve().parents[1] / "dexgrasp/test/physics.py"),
        git=subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True
        ).strip(),
        checkpoints={j["group"]: j["checkpoint_sha256"] for j in plan["jobs"]},
        candidates=n,
    )
    marker = output / "protocol.json"
    if marker.exists() and read(marker) != protocol:
        raise ValueError("Existing protocol differs")
    save(marker, protocol)
    chunks = output / "chunks"
    chunks.mkdir(exist_ok=True)
    outcomes = np.full((4, n), 255, dtype=np.uint8)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as pool:
        pending = {}
        for group, ids in jobs:
            path = chunks / f"{ids[0]}.npy"
            if path.exists():
                outcomes[:, ids] = np.load(path)
            else:
                pending[pool.submit(simulate, (group, ids))] = path
        for i, future in enumerate(as_completed(pending), 1):
            ids, result = future.result()
            path = pending[future]
            tmp = path.with_suffix(".tmp")
            with tmp.open("wb") as stream:
                np.save(stream, result)
            tmp.replace(path)
            outcomes[:, ids] = result
            if i % 100 == 0:
                print(i, "/", len(pending), flush=True)
    assert np.all(outcomes <= 2)
    np.save(output / "outcomes.npy", outcomes)
    report = {}
    for g, result in zip(models, outcomes):
        passed = result == 1
        aligned = models[g]["direction_pass"]
        count_objects = len(np.unique(objects))
        report[g] = dict(
            candidates=n,
            successes=int(passed.sum()),
            objects=count_objects,
            successful_objects=len(np.unique(objects[passed])),
            gsr=float(passed.mean()),
            osr=len(np.unique(objects[passed])) / count_objects,
            direction_pass=float(aligned.mean()),
            aligned_and_success=float((aligned & passed).mean()),
        )
    save(output / "summary.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", default="test")
    parser.add_argument("--workers", default=16, type=int)
    args = parser.parse_args()
    main(args.root, args.inputs or args.root, args.output, args.mode, args.workers)
