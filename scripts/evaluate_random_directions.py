"""Paired, uniform solid-angle commands over all held-out objects."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from dexgrasp.test.frozen import decode_shard, digest, init_decode, save


def indices(root, count, seed):
    rows = np.load(root / "direction_rows.npy")
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((len(rows) * count, 3))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.repeat(rows, count), np.repeat(np.arange(len(rows)), count), vectors.astype("f4")


def generate(root, out, count, seed, gpu, mode):
    import torch
    from dexgrasp.train.runner import load_model
    from dexgrasp.train.execution import configure_execution
    from dexgrasp.train.objectives import sample_keypoints

    torch.set_num_threads(1)
    group, rank = "ABCD"[gpu // 2], gpu % 2
    plan = json.loads((root / "plan.json").read_text())
    job = next(j for j in plan["jobs"] if j["group"] == group)
    if digest(job["checkpoint"]) != job["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    model, state = load_model(job["checkpoint"], device)
    configure_execution(model, dict(state["config"], compile_transformer=False))
    del state
    data = Path(plan["root"])
    points = np.load(data / "object_points.npy", mmap_mode="r")
    obs = np.load(data / "observation_index.npy", mmap_mode="r")
    packed, objects, directions = indices(root, count, seed)
    noise_ids = packed
    if mode == "noise":
        directions = np.load(data / "grasp_direction_cmd.npy", mmap_mode="r")[packed]
        noise_ids = seed + np.arange(len(packed))
    ids = np.arange(rank, len(packed), 2)
    folder = out / group / f"rank{rank}" / mode
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(
        folder / "index.npz",
        index=ids,
        packed=packed[ids],
        objects=objects[ids],
        directions=directions[ids],
    )
    predictions = []
    for offset in range(0, len(ids), 128):
        batch = ids[offset : offset + 128]
        condition = dict(
            object_points=torch.from_numpy(points[obs[packed[batch]]]).to(device),
            grasp_direction_cmd=torch.from_numpy(directions[batch]).to(device),
        )
        noise = torch.stack(
            [
                torch.randn((21, 3), generator=torch.Generator().manual_seed(plan["seed"] + int(r)))
                for r in noise_ids[batch]
            ]
        ).to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred = sample_keypoints(model, condition, steps=plan["sampling_steps"], noise=noise)
        predictions.append(pred.float().cpu().numpy() * 0.2)
    np.save(folder / "keypoints.npy", np.concatenate(predictions))


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    save(
        args.output / "design.json",
        dict(
            parent_sha256=digest(args.root / "plan.json"),
            seed=args.seed,
            directions_per_object=args.count,
            distribution=(
                "iid isotropic normal normalized to unit norm"
                if args.mode == "random"
                else "fixed observed direction"
            ),
            observation="frozen first test row per object",
            noise=(
                "shared by commands and models within object"
                if args.mode == "random"
                else "independent seeds per draw, shared across models"
            ),
            direction_threshold_deg=10,
            source=digest(__file__),
        ),
    )
    procs = [subprocess.Popen([sys.executable, *sys.argv, "--gpu", str(i)]) for i in range(8)]
    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError(f"GPU workers exited with {codes}")
    data = Path(json.loads((args.root / "plan.json").read_text())["root"])
    with ProcessPoolExecutor(
        args.workers, initializer=init_decode, initargs=(str(data / "hand/hand_spec.json"),)
    ) as pool:
        for folder in sorted(args.output.glob(f"*/rank*/{args.mode}")):
            decode_shard(folder, pool, chunk_size=16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=94117)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--mode", choices=("random", "noise"), default="random")
    args = parser.parse_args()
    if args.gpu is None:
        main(args)
    else:
        generate(args.root, args.output, args.count, args.seed, args.gpu, args.mode)
