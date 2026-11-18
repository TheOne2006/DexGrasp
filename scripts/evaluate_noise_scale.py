"""Rescale the same Gaussian draws at fixed observation and direction for a paired noise test."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dexgrasp.test.frozen import decode_shard, digest, init_decode, load_decoded_shard, save


def generate(job):
    import torch
    from dexgrasp.train.execution import configure_execution
    from dexgrasp.train.objectives import sample_keypoints
    from dexgrasp.train.runner import load_model

    group, gpu, args, indices = job
    torch.set_num_threads(1)
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    plan = json.loads((args.root / "plan.json").read_text())
    ckpt = next(row for row in plan["jobs"] if row["group"] == group)
    if digest(ckpt["checkpoint"]) != ckpt["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    model, state = load_model(ckpt["checkpoint"], device)
    configure_execution(model, dict(state["config"], compile_transformer=False))
    del state
    data = Path(plan["root"])
    cloud = np.load(data / "object_points.npy", mmap_mode="r")
    observation = np.load(data / "observation_index.npy", mmap_mode="r")
    original = json.loads((args.inputs / "design.json").read_text())
    for rank in (0, 1):
        folder = args.inputs / group / f"rank{rank}" / "noise"
        chosen = indices[rank::2]
        local = chosen // 2
        with np.load(folder / "index.npz") as src:
            np.testing.assert_array_equal(src["index"][local], chosen)
            packed, direction, obj = (src[k][local] for k in ("packed", "directions", "objects"))
        unit = np.load(folder / "keypoints.npy", mmap_mode="r")[local]
        for scale in args.scales:
            out = args.output / str(scale) / group / f"rank{rank}" / "noise"
            out.mkdir(parents=True, exist_ok=True)
            np.savez(
                out / "index.npz",
                index=np.arange(rank, len(indices), 2),
                packed=packed,
                directions=direction,
                objects=obj,
                original_index=chosen,
            )
            if scale == 1:
                keypoints = unit
            else:
                predicted = []
                for start in range(0, len(chosen), 128):
                    sl = slice(start, start + 128)
                    cond = dict(
                        object_points=torch.from_numpy(cloud[observation[packed[sl]]]).to(device),
                        grasp_direction_cmd=torch.from_numpy(direction[sl]).to(device),
                    )
                    noise = torch.stack(
                        [
                            torch.randn(
                                (21, 3),
                                generator=torch.Generator().manual_seed(
                                    plan["seed"] + original["seed"] + int(idx)
                                ),
                            )
                            for idx in chosen[sl]
                        ]
                    ).to(device)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        p = sample_keypoints(
                            model, cond, steps=plan["sampling_steps"], noise=noise * scale
                        )
                    predicted.append(p.float().cpu().numpy() * 0.2)
                keypoints = np.concatenate(predicted)
            np.save(out / "keypoints.npy", keypoints)
            if scale == 1:
                poses = load_decoded_shard(folder)[local]
                np.save(out / "decoded.npy", poses)
                save(
                    out / "decoded.done.json",
                    dict(count=len(poses), sha256=digest(out / "decoded.npy")),
                )


def main(args):
    from multiprocessing import get_context

    original = json.loads((args.inputs / "design.json").read_text())
    plan = json.loads((args.root / "plan.json").read_text())
    if digest(args.root / "plan.json") != original["parent_sha256"]:
        raise ValueError("Noise inputs belong to a different frozen plan")
    count = original["directions_per_object"]
    if not 0 < args.draws <= count or args.draws % 2 or count % 2:
        raise ValueError(
            "Paired two-shard sampling requires even counts and sufficient source draws"
        )
    if not 0 < args.objects <= len(plan["objects"]) or any(
        not np.isfinite(s) or s < 0 for s in args.scales
    ):
        raise ValueError("Invalid object count or noise scale")
    selected = np.sort(
        np.random.default_rng(args.seed).choice(len(plan["objects"]), args.objects, replace=False)
    )
    indices = (selected[:, None] * count + np.arange(args.draws)).ravel()
    args.output.mkdir(parents=True, exist_ok=True)
    for scale in args.scales:
        folder = args.output / str(scale)
        folder.mkdir(exist_ok=True)
        save(
            folder / "design.json",
            dict(
                source=digest(__file__),
                inputs=digest(args.inputs / "design.json"),
                scale=scale,
                directions_per_object=args.draws,
                object_indices=selected.tolist(),
                original_draw_indices=indices.tolist(),
                object_seed=args.seed,
                checkpoint_set=digest(args.root / "plan.json"),
            ),
        )
    jobs = [(g, args.devices[i % len(args.devices)], args, indices) for i, g in enumerate("ABCD")]
    with ProcessPoolExecutor(min(4, len(args.devices)), mp_context=get_context("spawn")) as pool:
        list(pool.map(generate, jobs))
    spec = str(Path(plan["root"]) / "hand/hand_spec.json")
    with ProcessPoolExecutor(args.workers, initializer=init_decode, initargs=(spec,)) as pool:
        for scale in args.scales:
            if scale == 1:
                continue
            for folder in sorted((args.output / str(scale)).glob("*/rank*/noise")):
                decode_shard(folder, pool, chunk_size=16)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "inputs", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--objects", type=int, default=64)
    p.add_argument("--draws", type=int, default=8)
    p.add_argument("--seed", type=int, default=191117)
    p.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    p.add_argument("--workers", type=int, default=16)
    main(p.parse_args())
