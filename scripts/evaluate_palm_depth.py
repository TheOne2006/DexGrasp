"""Recompute h32 on paired frozen GT and decoded grasps using original target triangles."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from dexgrasp.geometry.palm_depth import PalmSurfaceDepth
from dexgrasp.test.frozen import digest, save


def evaluate_group(job):
    group, indices = job
    geometry = groups[group]
    path = mesh_paths[geometry["asset_sha256"]]
    if digest(path) != geometry["asset_sha256"]:
        raise ValueError(f"Source mesh changed: {path}")
    with np.load(path) as asset:
        mesh = PalmSurfaceDepth(asset["vertices"], asset["faces"], geometry["scale"])
    values = np.full((len(poses), len(indices)), np.nan)
    for j, array in enumerate(poses.values()):
        rows = array[indices]
        for start in range(0, len(rows), 8):
            batch = rows[start : start + 8]
            valid = batch["decoded"]
            if valid.any():
                values[j, start : start + len(batch)][valid] = mesh.measure(
                    batch["R"][valid] @ rotation.T, batch["t"][valid]
                )
    return indices, values


def main(args):
    global poses, groups, mesh_paths, rotation
    src = args.internal
    target = args.output
    target.mkdir(parents=True, exist_ok=True)
    plan = json.loads((args.root / "plan.json").read_text())
    stage = json.loads((src / "plan.json").read_text())
    data = Path(plan["root"])
    labels = np.load(data / "palm_gap.npy", mmap_mode="r")[np.load(args.root / "test_rows.npy"), 0]
    rotation = np.array(
        json.loads(Path(stage["hand_spec"]).read_text())["official_from_grasp_rotation"]
    )
    poses = {g: np.load(src / (g + ".npy"), mmap_mode="r") for g in stage["models"]}
    groups = stage["groups"]
    gids = np.load(src / "groups.npy")
    ids = np.load(src / "paired.npy")
    sources = Path(__file__).resolve().parents[2] / "data/source_packages"
    mesh_paths = {}
    for package in plan["data_identity"]["source_identity"]:
        assets = json.loads((sources / package / "MANIFEST.json").read_text())["files"]
        for item in assets:
            if item["path"].startswith("assets/"):
                mesh_paths[item["sha256"]] = sources / package / item["path"]
    save(
        target / "protocol.json",
        {
            "internal_sha256": digest(src / "plan.json"),
            "paired_sha256": digest(src / "paired.npy"),
            "label": "palm-surface-nearest-mean32-grid4y8z-m-v1",
            "source": digest(__file__),
            "roles": list(poses),
            "size": len(ids),
            "selection": "10 paired test rows per object",
        },
    )
    actual = np.full((len(poses), len(ids)), np.nan)
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [(g, ids[gids[ids] == g]) for g in np.unique(gids[ids])]
        for indices, values in pool.map(evaluate_group, jobs):
            actual[:, np.searchsorted(ids, indices)] = values
    np.savez(target / "depths.npz", test_indices=ids, measured_m=actual, requested_m=labels[ids])
    report = {}
    for i, name in enumerate(poses):
        error = np.abs(actual[i] - labels[ids])
        report[name] = dict(
            count=len(error),
            measured=int(np.isfinite(error).sum()),
            mae_mm=float(np.nanmean(error) * 1000),
            median_mm=float(np.nanmedian(error) * 1000),
            p95_mm=float(np.nanquantile(error, 0.95) * 1000),
            max_mm=float(np.nanmax(error) * 1000),
            within_5mm=int((error <= 0.005).sum()),
        )
    save(target / "summary.json", report)
    np.testing.assert_allclose(actual[list(poses).index("GT")], labels[ids], atol=1e-6, rtol=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "internal", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    main(parser.parse_args())
