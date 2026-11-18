"""Prepare object-frame FK/IK poses, source GT and CoACD scenes for frozen self-test replay."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import trimesh

from dexgrasp.geometry.hand import URDFHand
from dexgrasp.test.frozen import DTYPE, decoded, digest, save


def read(path):
    return json.loads(path.read_text())


def ground_truth(data, packages, rows, centers, hand):
    gt = np.zeros(len(rows), dtype=DTYPE)
    gt["decoded"] = gt["fit"] = True
    index = read(data / "raw-index.json")
    keypoints = np.load(data / "keypoints_gt.npy", mmap_mode="r")
    offset = 0
    for shard in index["shards"]:
        end = offset + shard["count"]
        lo, hi = np.searchsorted(rows, [offset, end])
        if hi > lo:
            ids = rows[lo:hi] - offset
            path = packages / Path(shard["package"]).name / shard["shard"]
            with h5py.File(path) as archive:
                for key in ("R", "t", "q"):
                    gt[key][lo:hi] = archive[key][:][ids]
            for i in {lo, hi - 1}:
                pose = gt[i]
                expected = keypoints[rows[i]] * 0.2 + centers[i]
                np.testing.assert_allclose(
                    hand.world_keypoints(pose["R"], pose["t"], pose["q"]), expected, atol=1e-6
                )
        offset = end
    if offset != index["samples"]:
        raise ValueError("Incomplete packed corpus")
    return gt


def export_geometry(job):
    folder, pieces = job
    folder.mkdir(exist_ok=True, parents=True)
    model = ET.Element("mujoco")
    assets = ET.SubElement(model, "asset")
    world = ET.SubElement(model, "worldbody")
    volume = 0.0
    for i, piece in enumerate(pieces):
        volume += abs(trimesh.load_mesh(piece, process=False).volume)
        name = f"part_{i}"
        ET.SubElement(assets, "mesh", name=name, file=str(piece.resolve()))
        ET.SubElement(world, "geom", type="mesh", mesh=name)
    if volume <= 0:
        raise ValueError(f"Zero-volume collision geometry: {folder}")
    ET.ElementTree(model).write(folder / "coacd.xml")
    save(folder / "info.json", dict(scale=1, density=1, mass=volume))


def main(args):
    out = args.output
    out.mkdir(exist_ok=False, parents=True)
    parent = read(args.root / "plan.json")
    quality = read(args.root / "quality/protocol.json")
    data = Path(parent["root"])
    rows = np.load(args.root / "test_rows.npy")
    obs = np.load(args.root / "quality/test_observations.npy")
    idx = np.load(args.root / "object_indices.npy")
    np.save(out / "objects.npy", idx)
    centers = np.load(args.root / "quality/centers.npy")[obs]
    hand = URDFHand(quality["hand_spec"])
    for name in "ABCD":
        array = decoded(args.root, name)
        array["t"] += centers
        np.save(out / (name + ".npy"), array)
    np.save(out / "GT.npy", ground_truth(data, args.packages, rows, centers, hand))
    groups = quality["groups"]
    geometries = {g["asset_sha256"]: g for g in groups}
    jobs = []
    for sha, g in geometries.items():
        mesh_dir = args.collision_assets / sha
        manifest = read(mesh_dir / "manifest.json")
        pieces = [mesh_dir / Path(p).name for p in g["pieces"]]
        hashes = {Path(p).name: value for p, value in manifest["pieces"].items()}
        if any(digest(piece) != hashes[piece.name] for piece in pieces):
            raise ValueError(f"Collision geometry changed: {sha}")
        jobs.append((out / "assets" / sha, pieces))
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(export_geometry, jobs))
    lookup = {o: g["id"] for g in groups for o in g["observations"]}
    np.save(out / "groups.npy", np.array([lookup[o] for o in obs]))
    rng = np.random.default_rng(71183)
    paired = np.sort(
        np.concatenate(
            [rng.choice(np.flatnonzero(idx == i), 10, replace=False) for i in np.unique(idx)]
        )
    )
    np.save(out / "paired.npy", paired)
    names = ["GT", *"ABCD"]
    save(
        out / "plan.json",
        dict(
            parent=digest(args.root / "plan.json"),
            geometry=digest(args.root / "quality/protocol.json"),
            models=names,
            groups=groups,
            hand_spec=quality["hand_spec"],
            count=len(rows),
            paired=len(paired),
            object_count=len(parent["objects"]),
            seed=71183,
            source=digest(__file__),
            pose_files={name: digest(out / (name + ".npy")) for name in names},
            gt="Source R, t and q in original object frame, checked against packed FK by shard",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "output", "packages", "collision-assets"):
        parser.add_argument("--" + key, required=True, type=Path)
    main(parser.parse_args())
