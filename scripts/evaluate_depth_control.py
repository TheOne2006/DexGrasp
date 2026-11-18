"""Sweep h32 while fixing the input cloud, direction and flow noise for each test object."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from dexgrasp.geometry.palm_depth import PalmSurfaceDepth
from dexgrasp.test.frozen import digest, save
from dexgrasp.test.formal import official_hand
from dexgrasp.test.metrics import decode_metrics


def init_worker(hand_spec):
    global hand
    hand = official_hand(hand_spec)


def measure(job):
    mesh_path, mesh_sha, scale, direction, keypoints = job
    if digest(mesh_path) != mesh_sha:
        raise ValueError(f"Changed source mesh: {mesh_path}")
    with np.load(mesh_path) as geom:
        mesh = PalmSurfaceDepth(geom["vertices"], geom["faces"], scale)
    basis = np.asarray(hand.spec["official_from_grasp_rotation"])
    depths = []
    for k in keypoints:
        pose = decode_metrics(k, hand, direction)
        if pose["decoded"]:
            depths.append(mesh.measure(np.asarray(pose["R"]) @ basis.T, np.asarray(pose["t"])))
        else:
            depths.append(np.nan)
    return depths


def generate(checkpoint, points, directions, requested, seed, device):
    import torch
    from dexgrasp.test.inference import predict
    from dexgrasp.train.execution import configure_execution
    from dexgrasp.train.runner import load_model

    model, state = load_model(checkpoint, device)
    configure_execution(model, dict(state["config"], compile_transformer=False))
    del state
    rng = torch.Generator().manual_seed(seed)
    noise = torch.randn(len(points), 21, 3, generator=rng)
    grid = requested if model.cfg.use_gap else requested[:1]
    draws = []
    for h in grid:
        batches = []
        for start in range(0, len(points), 64):
            part = slice(start, start + 64)
            condition = dict(object_points=points[part], grasp_direction_cmd=directions[part])
            if model.cfg.use_gap:
                condition["palm_gap"] = np.full((len(points[part]), 1), h, dtype=np.float32)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                k = predict(model, condition, steps=20, noise=noise[part])
            batches.append(k[:, 0].cpu().numpy())
        draws.append(np.concatenate(batches))
    return np.stack(draws, axis=1)


def main(args):
    import multiprocessing
    import torch

    target = args.output
    target.mkdir(parents=True, exist_ok=True)
    parent = json.loads((args.root / "plan.json").read_text())
    data = Path(parent["root"])
    rows = np.load(args.root / "direction_rows.npy")
    obs = np.load(data / "observation_index.npy", mmap_mode="r")[rows]
    centers = np.load(args.root / "quality/centers.npy")[obs]
    points = np.load(data / "object_points.npy", mmap_mode="r")[obs] * 0.2 + centers[:, None]
    directions = np.load(data / "grasp_direction_cmd.npy", mmap_mode="r")[rows]
    train = np.load(data / "train_rows.npy")
    labels = np.load(data / "palm_gap.npy", mmap_mode="r")[train, 0]
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    requested = np.quantile(labels, quantiles)
    quality = json.loads((args.root / "quality/protocol.json").read_text())
    group_by_obs = {i: g for g in quality["groups"] for i in g["observations"]}
    sources = Path(__file__).resolve().parents[2] / "data/source_packages"
    meshes = {}
    for src in parent["data_identity"]["source_identity"]:
        for item in json.loads((sources / src / "MANIFEST.json").read_text())["files"]:
            if item["path"].startswith("assets/"):
                meshes[item["sha256"]] = sources / src / item["path"]
    save(
        target / "protocol.json",
        dict(
            checkpoint=digest(args.checkpoint),
            seed=args.seed,
            quantiles=quantiles,
            h_m=requested.tolist(),
            directions=digest(args.root / "direction_rows.npy"),
            source=digest(__file__),
            parent=digest(args.root / "plan.json"),
            hand=digest(quality["hand_spec"]),
            design="One observation per held-out object, all depth commands use identical noise and direction",
        ),
    )
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    keypoints = generate(args.checkpoint, points, directions, requested, args.seed, device)
    jobs = []
    for i, oid in enumerate(obs):
        group = group_by_obs[oid]
        sha = group["asset_sha256"]
        jobs.append((meshes[sha], sha, group["scale"], directions[i], keypoints[i]))
    with ProcessPoolExecutor(
        args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker,
        initargs=(quality["hand_spec"],),
    ) as pool:
        achieved = np.array(list(pool.map(measure, jobs)))
    if achieved.shape[1] == 1:
        achieved = np.repeat(achieved, len(requested), axis=1)
    np.savez(target / "sweep.npz", rows=rows, requested_m=requested, achieved_m=achieved)
    valid_pairs = np.isfinite(achieved[:, [0, -1]]).all(axis=1)
    span = achieved[valid_pairs, -1] - achieved[valid_pairs, 0]
    error = np.abs(achieved - requested)
    save(
        target / "summary.json",
        dict(
            objects=len(points),
            measured=np.isfinite(achieved).sum(axis=0).tolist(),
            requested_mm=(requested * 1000).tolist(),
            achieved_median_mm=(np.nanmedian(achieved, axis=0) * 1000).tolist(),
            mae_mm=(np.nanmean(error, axis=0) * 1000).tolist(),
            within_5mm_rate=(error <= 0.005).mean(axis=0).tolist(),
            endpoint_pairs=int(valid_pairs.sum()),
            mean_endpoint_shift_mm=float(span.mean() * 1000),
            increasing_endpoint_rate=float((span > 0).mean()),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "checkpoint", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--seed", type=int, default=19117)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    main(parser.parse_args())
