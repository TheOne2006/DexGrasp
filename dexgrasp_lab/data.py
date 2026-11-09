"""Object-level splits and lazy adapters; coordinates are in metres."""
import hashlib
import json
from pathlib import Path

REPRESENTATION = "shadow28_xyz_m_3phase"
PHASES = ("pregrasp", "grasp", "squeeze")
SPLITS = ("train", "val", "test")


def normalize_split(split):
    """Accept the earlier evaluation alias; use test in project interfaces."""
    return "test" if split == "evaluation" else split


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def split_objects(train, evaluation, fraction=0.1, seed=0):
    if not 0 < fraction < 1:
        raise ValueError("validation fraction must be in (0, 1)")
    for ids in (train, evaluation):
        if len(set(ids)) != len(ids) or not all(isinstance(x, str) and x for x in ids):
            raise ValueError("object IDs must be unique nonempty strings")
    if set(train) & set(evaluation):
        raise ValueError("official train/test object leakage")
    if len(train) < 2 or not evaluation:
        raise ValueError("need >=2 training objects and >=1 test object")
    order = sorted(train, key=lambda x: hashlib.sha256(f"{seed}:{x}".encode()).hexdigest())
    n_val = max(1, min(len(train) - 1, round(len(train) * fraction)))
    return {**{x: "val" for x in order[:n_val]},
            **{x: "train" for x in order[n_val:]}, **{x: "test" for x in evaluation}}


def load_manifest(path, check_files=True):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("representation") != REPRESENTATION:
        raise ValueError("unsupported manifest schema or representation")
    records = manifest["records"]
    if not records:
        raise ValueError("empty manifest")
    seen, splits, npz_splits, h5_ranges = set(), {}, {}, {}
    for row in records:
        row["split"] = normalize_split(row["split"])
        sid, oid, split = row["id"], row["object_id"], row["split"]
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError("duplicate/invalid record ID")
        seen.add(sid)
        if not isinstance(oid, str) or not oid or split not in SPLITS:
            raise ValueError("invalid object ID or split")
        if oid in splits and splits[oid] != split:
            raise ValueError(f"object leakage: {oid}")
        splits[oid] = split
        storage = row["storage"]
        fields = {"npz": ["path"], "kpgrasp_h5": ["grasp_h5", "point_cloud_h5"]}
        if storage not in fields:
            raise ValueError(f"unknown storage: {storage}")
        if storage == "kpgrasp_h5":
            if any(type(row[k]) is not int for k in ("offset", "count", "pc_index")):
                raise ValueError("HDF5 offsets/count must be integers")
            if row["offset"] < 0 or row["count"] < 1 or row["pc_index"] < 0:
                raise ValueError("invalid HDF5 offset/count")
        for key in fields[storage]:
            location = Path(row[key]).expanduser()
            row[key] = str((path.parent / location).resolve())
            if check_files and not Path(row[key]).is_file():
                raise FileNotFoundError(row[key])
        if row.get("scene_path"):
            row["scene_path"] = str((path.parent / row["scene_path"]).resolve())
        if storage == "npz":
            previous = npz_splits.setdefault(row["path"], split)
            if previous != split:
                raise ValueError("same NPZ file appears in different splits")
        else:
            ranges = h5_ranges.setdefault(row["grasp_h5"], [])
            start, end = row["offset"], row["offset"] + row["count"]
            for other_start, other_end, other_split in ranges:
                if split != other_split and max(start, other_start) < min(end, other_end):
                    raise ValueError("overlapping HDF5 rows across splits")
            ranges.append((start, end, split))
    if set(splits.values()) != set(SPLITS):
        raise ValueError("manifest must contain nonempty train/val/test splits")
    return manifest


def build_kp_manifest(root, output, fraction=0.1, seed=0):
    """Use official object splits; never repurpose official test as validation."""
    import h5py
    root = Path(root).resolve()
    index_path = root / "dex_kp_h5/index_obj_grasp.json"
    index = json.loads(index_path.read_text())
    records, source_hashes, global_ids = [], {str(index_path): digest(index_path)}, set()
    for collection in ("DGN_5k", "objaverse_5k"):
        folder = root / "object" / collection
        split_files = [folder / "valid_split" / f"{s}.json" for s in ("train", "test")]
        train, evaluation = [json.loads(p.read_text()) for p in split_files]
        source_hashes.update({str(p): digest(p) for p in split_files})
        assignments = split_objects(train, evaluation, fraction, seed)
        if global_ids & set(assignments):
            raise ValueError("cross-collection object IDs collide in upstream grasp index")
        global_ids.update(assignments)
        pc_file = folder / "vision_data/compl_pc.h5"
        with h5py.File(pc_file, "r") as f:
            names = [x.decode() if isinstance(x, bytes) else str(x) for x in f["filenames"][:]]
            if len(set(names)) != len(names) or len(names) != len(f["data"]):
                raise ValueError("duplicate/misaligned point-cloud IDs")
            pc_ids = {name: i for i, name in enumerate(names)}
        for oid, split in sorted(assignments.items()):
            # Missing objects are errors, not silently removed from evaluation.
            offset, count, shard = index[oid]
            records.append({"id": f"{collection}/{oid}", "object_id": f"{collection}/{oid}",
                            "split": split, "storage": "kpgrasp_h5",
                            "grasp_h5": str(root / "dex_kp_h5" / shard),
                            "offset": offset, "count": count, "pc_index": pc_ids[oid],
                            "point_cloud_h5": str(pc_file)})
    manifest = {"schema_version": 1, "representation": REPRESENTATION,
                "dataset_id": "dexonomy_kp", "split_seed": seed,
                "validation_fraction": fraction, "source_hashes": source_hashes,
                "records": records}
    # Validate before publishing a manifest; no output on failure.
    validate_records(records)
    write_json(output, manifest)
    return manifest


def read_record(row, rng, targets=True):
    import numpy as np
    if row["storage"] == "npz":
        with np.load(row["path"], allow_pickle=False) as f:
            pc = f["point_cloud"].astype(np.float32)
            kp = None
            if targets:
                kps = f["keypoints"]
                if kps.ndim != 4 or kps.shape[1:] != (3, 28, 3) or len(kps) == 0:
                    raise ValueError("keypoints must be (G, 3, 28, 3), G>0")
                kp = kps[int(rng.integers(len(kps)))].astype(np.float32)
    else:
        import h5py
        idx = row["offset"] + int(rng.integers(row["count"]))
        with h5py.File(row["grasp_h5"], "r") as f:
            kp = np.stack([f[f"{p}_kp"][idx] for p in PHASES]).astype(np.float32)
            scale = np.asarray(f["scene_scale"][idx]).reshape(-1)
            if scale.size != 1 or not np.isfinite(scale).all() or scale[0] <= 0:
                raise ValueError("scene_scale must be a positive scalar")
        with h5py.File(row["point_cloud_h5"], "r") as f:
            pc = np.asarray(f["data"][row["pc_index"]], dtype=np.float32) * scale[0]
    if pc.ndim != 2 or pc.shape[1] != 3 or len(pc) == 0 or (kp is not None and kp.shape != (3, 28, 3)):
        raise ValueError("invalid point-cloud or keypoint shape")
    if not np.isfinite(pc).all() or (kp is not None and not np.isfinite(kp).all()):
        raise ValueError("nonfinite coordinates")
    return pc, kp


def validate_records(records):
    """Check all array shapes/bounds and one deterministic example per record."""
    import numpy as np
    for row in records:
        if row["storage"] == "kpgrasp_h5":
            import h5py
            with h5py.File(row["grasp_h5"], "r") as f:
                for key in [f"{p}_kp" for p in PHASES] + ["scene_scale"]:
                    if row["offset"] + row["count"] > len(f[key]):
                        raise ValueError("grasp index exceeds shard bounds")
                    if key.endswith("_kp") and f[key].shape[1:] != (28, 3):
                        raise ValueError("wrong keypoint convention")
        read_record(row, np.random.default_rng(0), targets=normalize_split(row["split"]) != "test")


def batch(records, indices, rng, num_points, scale_m, device, targets=True):
    import numpy as np
    import torch
    clouds, keypoints, centres = [], [], []
    for i in indices:
        pc, kp = read_record(records[int(i)], rng, targets=targets)
        centre = pc.mean(axis=0)
        selection = rng.choice(len(pc), num_points, replace=len(pc) < num_points)
        clouds.append((pc[selection] - centre) / scale_m)
        if targets:
            keypoints.append((kp - centre) / scale_m)
        centres.append(centre)
    return (torch.as_tensor(np.stack(clouds), device=device),
            torch.as_tensor(np.stack(keypoints), device=device) if targets else None, np.stack(centres))
