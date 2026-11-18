"""Bind frozen direction and noise draws to the original object pose and native force task."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
from dexgrasp.test.frozen import decoded, digest, save


def read_draws(folder, mode):
    """Interleave the two frozen inference ranks, validating identities for all four models."""
    reference = {}
    for group in "ABCD":
        for rank in range(2):
            with np.load(folder / group / f"rank{rank}" / mode / "index.npz") as index:
                for field in ("index", "packed", "objects", "directions"):
                    key = rank, field
                    if group == "A":
                        reference[key] = index[field]
                    else:
                        np.testing.assert_array_equal(reference[key], index[field])
    order = np.argsort(np.concatenate([reference[r, "index"] for r in range(2)]))
    np.testing.assert_array_equal(
        np.concatenate([reference[r, "index"] for r in range(2)])[order], np.arange(len(order))
    )
    return [np.concatenate([reference[r, field] for r in range(2)])[order]
            for field in ("packed", "objects")]


def prepare(args):
    parent = json.loads((args.root / "plan.json").read_text())
    source = json.loads((args.source / "plan.json").read_text())
    test = np.load(args.root / "test_rows.npy")
    assert source["selection"] == "all" and source["selected"] == digest(args.root / "test_rows.npy")
    observation = np.load(Path(parent["root"]) / "observation_index.npy", mmap_mode="r")
    centers = np.load(args.root / "quality/centers.npy", mmap_mode="r")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "plan.json").exists():
        raise FileExistsError("A frozen plan already exists in the output directory")
    poses = defaultdict(list)
    settings = {}
    originals = []
    object_ids = []
    offset = 0
    for name, mode, path in args.draws:
        if name in settings or mode not in ("random", "noise"):
            raise ValueError(f"Duplicate draw set or unsupported mode: {name}, {mode}")
        folder = Path(path).resolve()
        packed, objects = read_draws(folder, mode)
        ix = np.searchsorted(test, packed)
        np.testing.assert_array_equal(test[ix], packed)
        for group in "ABCD":
            values = decoded(folder, group, mode)
            assert len(values) == len(packed)
            values["t"] += centers[observation[packed]]
            poses[group].append(values)
        settings[name] = dict(start=offset, stop=offset + len(ix), mode=mode, inputs=str(folder),
                              design=digest(folder / "design.json"))
        originals.extend(ix.tolist())
        object_ids.extend(objects.tolist())
        offset += len(ix)
    wanted = set(originals)
    selected = {row["index"]: (sid, row) for sid, rows in source["sources"].items()
                for row in rows if row["index"] in wanted}
    assert set(selected) == wanted
    scenes = args.output / "scenes"
    scenes.mkdir(exist_ok=True)
    records = defaultdict(list)
    for index, (old, obj) in enumerate(zip(originals, object_ids, strict=True)):
        sid, row = selected[old]
        assert row["object"] == obj
        records[sid].append(dict(index=index, object=obj, mode=row["mode"]))
    for sid in records:
        link = scenes / f"{sid}.npy"
        path = (args.source / "scenes" / link.name).resolve(strict=True)
        if link.exists():
            assert link.resolve() == path
        else:
            link.symlink_to(path)
    hashes = {}
    for group, chunks in poses.items():
        path = args.output / f"{group}.npy"
        np.save(path, np.concatenate(chunks))
        hashes[group] = digest(path)
    np.save(args.output / "source_rows.npy", np.array(originals, dtype=np.int64))
    np.save(args.output / "objects.npy", np.array(object_ids, dtype=np.int32))
    plan = {k: source[k] for k in ("objects", "spec", "hand", "hand_sha", "resident_source",
                                    "cpu_adapter", "audit", "code", "task")}
    plan.update(stages=str(args.output.resolve()), groups=list("ABCD"), sources=records, poses=hashes,
                sets=settings, source_plan=digest(args.source / "plan.json"), builder=digest(__file__),
                source_rows=digest(args.output / "source_rows.npy"),
                object_map=digest(args.output / "objects.npy"),
                selection="conditioned draws sharing the archived observation and original scene")
    save(args.output / "plan.json", plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "source", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--draws", nargs=3, action="append", metavar=("NAME", "MODE", "FOLDER"),
                        required=True)
    prepare(parser.parse_args())
