"""Audit every held-out GT against Weilun's recorded Floating FC or Tabletop Slide+FC receipt."""

import argparse
import base64
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import json
from pathlib import Path

import h5py
import numpy as np

from dexgrasp.test.frozen import digest, save


AXES = {tuple(v * sign) for v in np.eye(3, dtype=int) for sign in (1, -1)}


def unpack(value):
    if isinstance(value, dict):
        return np.frombuffer(base64.b64decode(value["__ndarray__"]), dtype=value["dtype"]).reshape(
            value["shape"]
        )
    return np.asarray(value)


def physics_verified(metric):
    if "direction_results" in metric:
        dirs = metric["direction_results"]
        if any(
            [
                metric.get("type") != "force_closure_6d",
                metric.get("passes") is not True,
                metric.get("directions_completed") != 6,
                metric.get("directions_required") != 6,
                metric.get("translation_threshold_m") != 0.05,
                metric.get("rotation_threshold_deg") != 15,
            ]
        ):
            return False
        checks = all(
            d["passes"]
            and d["completed_holds"] == 10
            and 0 <= d["max_translation_error_m"] < 0.05
            and 0 <= d["max_rotation_error_deg"] < 15
            for d in dirs
        )
    else:
        dirs = metric["directions"]
        slide = metric["slide"]
        if not (
            metric.get("joint_success") is True
            and slide["passes"]
            and 0 <= slide["translation_m"] < 0.05
            and 0 <= slide["rotation_deg"] < 15
        ):
            return False
        snaps = {d.get("start_snapshot_sha256") for d in dirs}
        checks = (
            len(snaps) == 1
            and all(isinstance(v, str) and len(v) == 64 for v in snaps)
            and all(
                d["candidate_id"] == slide["candidate_id"]
                and d["passes"]
                and not unpack(d["table_contact_counts"]).any()
                and len(d["pose_checks"]) == 10
                and all(
                    c["passes"] and 0 <= c["translation_m"] < 0.05 and 0 <= c["rotation_deg"] < 15
                    for c in d["pose_checks"]
                )
                for d in dirs
            )
        )
    return bool(checks and len(dirs) == 6 and {tuple(unpack(d["direction"])) for d in dirs} == AXES)


def check_shard(job):
    path, lo, indices = job
    result = []
    profiles = Counter()
    with h5py.File(path) as f:
        flags = f["source_success"][:][indices]
        bounds = f["metrics_offsets"][:]
        raw = f["metrics_bytes"][:]
        provenance_offsets = f["provenance_offsets"][:]
        provenance = f["provenance_bytes"][:]
        for idx, flag in zip(indices, flags):
            metric = json.loads(bytes(raw[bounds[idx] : bounds[idx + 1]]))
            source = json.loads(
                bytes(provenance[provenance_offsets[idx] : provenance_offsets[idx + 1]])
            )
            mode = "floating_fc" if "direction_results" in metric else "tabletop_slide_fc"
            profiles[
                (
                    mode,
                    metric.get("backend", ""),
                    metric.get("physics_profile", metric.get("protocol", "")),
                )
            ] += 1
            direct = bool(
                source.get("eval_path")
                and isinstance(source.get("result_index"), int)
                and source["result_index"] >= 0
            )
            snapshot = bool(
                source.get("eval_locator")
                and source.get("snapshot_shard")
                and len(source.get("snapshot_shard_sha256", "")) == 64
                and isinstance(source.get("snapshot_row"), int)
                and source["snapshot_row"] >= 0
            )
            linked = len(source.get("eval_sha256", "")) == 64 and (direct or snapshot)
            result.append(
                (mode == "tabletop_slide_fc", bool(flag), physics_verified(metric), linked)
            )
    return lo, np.asarray(result, dtype=bool), profiles


def main(args):
    root = args.root
    out = args.output
    out.mkdir(exist_ok=True, parents=True)
    plan = json.loads((root / "plan.json").read_text())
    data = Path(plan["root"])
    split = np.load(root / "test_rows.npy")
    objects = np.load(root / "object_indices.npy")
    index = json.loads((data / "raw-index.json").read_text())
    save(
        out / "protocol.json",
        dict(
            source=digest(__file__),
            split=digest(root / "test_rows.npy"),
            row_count=len(split),
            object_count=len(np.unique(objects)),
            parent=digest(root / "plan.json"),
            check="Archived flag and six distinct axes, ten hold checks per axis with displacement <5 cm and rotation <15 deg. Tabletop also requires Slide pass, common released snapshot SHA and no force-phase table contact.",
            scope="Archived recorded outcomes from original production, no new simulation",
        ),
    )
    jobs = []
    offset = 0
    for s in index["shards"]:
        end = offset + s["count"]
        a, b = np.searchsorted(split, [offset, end])
        if b > a:
            jobs.append(
                (args.packages / Path(s["package"]).name / s["shard"], a, split[a:b] - offset)
            )
        offset = end
    if offset != index["samples"]:
        raise ValueError("Incomplete packed index")
    checks = np.empty((len(split), 4), bool)
    seen = np.zeros(len(split), bool)
    tallies = Counter()
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(check_shard, j) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            a, values, profiles = f.result()
            checks[a : a + len(values)] = values
            seen[a : a + len(values)] = True
            tallies.update(profiles)
            if i % 200 == 0:
                print(i, len(futures), flush=True)
    assert seen.all()
    np.savez(
        out / "row_audit.npz",
        test_rows=split,
        tabletop=checks[:, 0],
        source_success=checks[:, 1],
        six_axis_verified=checks[:, 2],
        source_link=checks[:, 3],
    )
    summary = {}
    for name, mask in [
        ("all", seen),
        ("floating_fc", ~checks[:, 0]),
        ("tabletop_slide_fc", checks[:, 0]),
    ]:
        total = int(mask.sum())
        passed = mask & checks[:, 1] & checks[:, 2]
        nobj = len(np.unique(objects[mask]))
        sobj = len(np.unique(objects[passed]))
        summary[name] = dict(
            candidates=total,
            source_success=int(checks[mask, 1].sum()),
            six_axis_verified=int(checks[mask, 2].sum()),
            successes=int(passed.sum()),
            source_links=int(checks[mask, 3].sum()),
            objects=nobj,
            successful_objects=sobj,
            gsr=float(passed.sum() / total),
            osr=sobj / nobj,
        )
    summary["profiles"] = {" / ".join(k): v for k, v in tallies.items()}
    save(out / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "output", "packages"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--workers", type=int, default=24)
    main(parser.parse_args())
