"""Prepare the official floating scenes, then infer and decode Sharpa grasps."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
from dexgrasp.data.source_objects import observation_objects, split_objects
from dexgrasp.test.frozen import decode_shard, digest, init_decode, save


def prepare(root, assets, out, seed, per_dataset, allow_mesh_clouds=False):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "plan.json").exists():
        raise FileExistsError(f"Prepared protocol already exists in {out}")
    parent = json.loads((root / "plan.json").read_text())
    train = set(split_objects(parent["root"], parent["data_identity"], "train"))
    rng = np.random.default_rng(seed)
    scenes, coverage = [], {}
    clouds = []
    for dataset in ("DGN_5k", "objaverse_5k"):
        base = assets / dataset
        split = base / "valid_split/test.json"
        ids = sorted(json.loads(split.read_text()))
        selected = (
            ids
            if per_dataset is None
            else sorted(rng.choice(ids, min(len(ids), per_dataset), replace=False))
        )
        coverage[dataset] = dict(
            split_sha256=digest(split),
            objects=len(ids),
            overlap=sum(i in train for i in ids),
            selected=len(selected),
        )
        missing_scenes = []
        cloud_sources = {}
        for obj in selected:
            path = base / "vision_data/complete_point_cloud" / obj / "pc.npy"
            entries = sorted((base / "scene_cfg" / obj / "floating").glob("*.npy"))
            if not entries:
                missing_scenes.append(obj)
                continue
            if path.exists():
                cloud = np.load(path)
                source = "released complete point cloud"
            elif allow_mesh_clouds:
                from dexgrasp.geometry.mesh import load_mesh, sample_surface

                path = base / "processed_data" / obj / "mesh/simplified.obj"
                cloud = sample_surface(*load_mesh(path), 65536, rng)
                source = "65536 area-weighted surface samples from released simplified.obj"
            else:
                raise FileNotFoundError(path)
            cloud_sources[obj] = {"kind": source, "source_sha256": digest(path)}

            if cloud.ndim != 2 or cloud.shape[1] != 3 or not np.isfinite(cloud).all():
                raise ValueError(f"Invalid official point cloud: {path}")
            for entry in entries:
                meta = np.load(entry, allow_pickle=True).item()
                assert meta["task"]["obj_name"] == obj and meta["task"]["type"] == "force_closure"
                geom = meta["scene"][obj]
                np.testing.assert_allclose(geom["pose"], [0, 0, 0, 1, 0, 0, 0])
                pts = cloud[rng.choice(len(cloud), 1024, replace=True)] * geom["scale"]
                clouds.append(pts.astype("f4"))
                scenes.append(
                    dict(
                        object=obj,
                        source=dataset,
                        training_overlap=obj in train,
                        path=str(entry.resolve()),
                    )
                )
        coverage[dataset]["point_cloud_inputs"] = cloud_sources
        coverage[dataset]["listed_without_floating_scene"] = missing_scenes
        coverage[dataset]["evaluated_objects"] = len(selected) - len(missing_scenes)
    directions = rng.normal(size=(len(scenes) * 10, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    np.save(out / "directions.npy", directions.astype("f4"))
    np.save(out / "points.npy", np.stack(clouds))
    save(
        out / "plan.json",
        dict(
            scenes=scenes,
            assets_root=str(assets.resolve()),
            asset_manifest=digest(assets / "READY.json"),
            coverage=coverage,
            seed=seed,
            per_dataset=per_dataset,
            candidates_per_scene=10,
            selection="10 independent noise and uniform sphere direction draws, unranked",
            source=digest(__file__),
            parent_plan_sha256=digest(root / "plan.json"),
            jobs=parent["jobs"],
            hand_spec=str(Path(parent["root"]) / "hand/hand_spec.json"),
            points_sha256=digest(out / "points.npy"),
            directions_sha256=digest(out / "directions.npy"),
        ),
    )


def audit_mesh_overlap(root, assets, packages, out):
    parent = json.loads((root / "plan.json").read_text())
    plan = json.loads((out / "plan.json").read_text())
    if digest(root / "plan.json") != plan["parent_plan_sha256"]:
        raise ValueError("Official evaluation and training inventory use different frozen plans")
    data = Path(parent["root"])
    rows = np.load(data / "train_rows.npy", mmap_mode="r")
    indices = np.unique(np.load(data / "observation_index.npy", mmap_mode="r")[rows])
    labels = observation_objects(packages, parent["data_identity"], indices)
    train = set(split_objects(data, parent["data_identity"], "train"))
    if {name for name, _ in labels} != train:
        raise ValueError("Training observations and frozen object split disagree")
    by_mesh = defaultdict(set)
    for name, mesh in labels:
        by_mesh[mesh].add(name)
    objects, summary = {}, {}
    for source, dataset in plan["coverage"].items():
        records = {}
        for obj in dataset["point_cloud_inputs"]:
            mesh = digest(assets / source / "processed_data" / obj / "mesh/simplified.obj")
            records[obj] = dict(
                official_mesh_sha256=mesh,
                id_overlap=obj in train,
                sha_overlap=mesh in by_mesh,
                matching_train_ids=sorted(by_mesh.get(mesh, ())),
            )
        objects[source] = records
        summary[source] = dict(
            total=len(records),
            id_overlap=sum(r["id_overlap"] for r in records.values()),
            sha_overlap=sum(r["sha_overlap"] for r in records.values()),
            unseen_id_and_mesh=sum(
                not (r["sha_overlap"] or r["id_overlap"]) for r in records.values()
            ),
        )
    save(
        out / "mesh_overlap.json",
        dict(
            train_objects=len(train),
            train_observations=len(indices),
            unique_train_source_mesh_hashes=len(by_mesh),
            summary=summary,
            objects=objects,
        ),
    )


def generate(out, gpu, device_index):
    import torch
    from dexgrasp.test.inference import predict
    from dexgrasp.train.runner import load_model
    from dexgrasp.train.execution import configure_execution

    plan = json.loads((out / "plan.json").read_text())
    group, rank = "ABCD"[gpu // 2], gpu % 2
    job = next(j for j in plan["jobs"] if j["group"] == group)
    assert digest(job["checkpoint"]) == job["checkpoint_sha256"]
    device = torch.device(f"cuda:{gpu if device_index is None else device_index}")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    model, state = load_model(job["checkpoint"], device)
    configure_execution(model, dict(state["config"], compile_transformer=False))
    del state
    points = np.load(out / "points.npy", mmap_mode="r")
    dirs = np.load(out / "directions.npy")
    ids = np.arange(rank, len(dirs), 2)
    folder = out / group / f"rank{rank}" / "official"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / "index.npz", index=ids, directions=dirs[ids])
    result = np.lib.format.open_memmap(
        folder / "keypoints.npy", mode="w+", dtype="f4", shape=(len(ids), 21, 3)
    )
    for offset in range(0, len(ids), 256):
        batch = ids[offset : offset + 256]
        noise = torch.stack(
            [
                torch.randn(21, 3, generator=torch.Generator().manual_seed(plan["seed"] + int(i)))
                for i in batch
            ]
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            values = predict(
                model,
                dict(object_points=points[batch // 10], grasp_direction_cmd=dirs[batch]),
                steps=20,
                noise=noise,
            )
        result[offset : offset + len(batch)] = values[:, 0].float().cpu().numpy()
    result.flush()


def decode(out, workers):
    plan = json.loads((out / "plan.json").read_text())
    with ProcessPoolExecutor(
        workers, initializer=init_decode, initargs=(plan["hand_spec"],)
    ) as pool:
        for folder in sorted(out.glob("*/rank*/official")):
            decode_shard(folder, pool, chunk_size=64)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "generate", "decode", "audit-mesh"))
    p.add_argument("--root", type=Path)
    p.add_argument("--assets", type=Path)
    p.add_argument("--packages", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=71819)
    p.add_argument(
        "--allow-mesh-clouds",
        action="store_true",
        help="Reconstruct absent complete clouds on official meshes and record each source",
    )
    p.add_argument("--per-dataset", type=int)
    p.add_argument("--gpu", type=int)
    p.add_argument("--device-index", type=int)
    p.add_argument("--workers", type=int, default=64)
    args = p.parse_args()
    if args.action == "prepare":
        prepare(
            args.root, args.assets, args.output, args.seed, args.per_dataset, args.allow_mesh_clouds
        )
    elif args.action == "generate":
        generate(args.output, args.gpu, args.device_index)
    elif args.action == "audit-mesh":
        if not all((args.root, args.assets, args.packages)):
            p.error("audit-mesh requires --root, --assets and --packages")
        audit_mesh_overlap(args.root, args.assets, args.packages, args.output)
    else:
        decode(args.output, args.workers)
