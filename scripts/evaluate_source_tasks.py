"""Replay a balanced self-test in each row's original world frame, task and native solver."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import numpy as np

repo = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(repo / "simulation"), str(repo)]
from dexgrasp.test.frozen import digest, save


def load(path):
    return json.loads(Path(path).read_text())


def convert_object(job):
    from dexonomy.upstream import SCHEMA_VERSION, convert_request
    from dexonomy.util.file_util import load_scene_cfg

    name, xml, out = job
    folder = out / "assets" / name
    folder.mkdir(parents=True, exist_ok=True)
    pieces = sorted((xml.parent / "meshes").glob("*.obj"))
    surface = xml.parent.parent / "mesh/simplified.obj"
    request = dict(
        schema_version=SCHEMA_VERSION,
        request_id=name,
        world_frame="world",
        units=dict(length="m", mass="kg", time="s", angle="rad"),
        scene=dict(
            entities=[
                dict(
                    id=name,
                    role="target",
                    type="rigid_mesh",
                    scale=[1.0, 1.0, 1.0],
                    pose=dict(position=[0.0, 0.0, 0.0], quaternion_wxyz=[1.0, 0.0, 0.0, 0.0]),
                    physical=dict(mass_kg=0.1),
                    assets=dict(surface_mesh=str(surface), collision_meshes=list(map(str, pieces))),
                )
            ]
        ),
    )
    save(folder / "request.json", request)
    converted = convert_request(folder / "request.json", folder / "converted", overwrite=True)
    obj = load_scene_cfg(str(converted["stability_scene"]))["scene"][name]
    return name, {k: v for k, v in obj.items() if k.endswith("_path")}


def prepare(args):
    import h5py

    root, stages, out = args.root, args.stages, args.output
    out.mkdir(parents=True, exist_ok=True)
    parent = load(root / "plan.json")
    frozen = load(stages / "plan.json")
    rows = np.load(root / "test_rows.npy")
    ids = np.arange(len(rows)) if args.all_rows else np.load(stages / "paired.npy")
    geometries = np.load(stages / "groups.npy")
    objects = np.load(stages / "objects.npy")
    raw = np.load(args.audit / "row_audit.npz")
    np.testing.assert_array_equal(rows, raw["test_rows"])
    tabletop = raw["tabletop"][ids]
    selected = rows[ids]
    assert np.all(np.diff(selected) > 0)
    if not args.all_rows:
        assert np.all(np.bincount(objects[ids]) == 10)
    official = load(args.audit / "official_self_assets.json")["assets"]
    jobs = [(name, Path(official[name]["physics_xml"]), out) for name in parent["objects"]]
    with ProcessPoolExecutor(args.workers) as pool:
        asset = dict(pool.map(convert_object, jobs))
    records = defaultdict(list)
    scenes = {}
    offset = 0
    for shard in load(Path(parent["root"]) / "raw-index.json")["shards"]:
        end = offset + shard["count"]
        a, b = np.searchsorted(selected, [offset, end])
        if a < b:
            path = args.packages / Path(shard["package"]).name / shard["shard"]
            inside = selected[a:b] - offset
            with h5py.File(path) as f:
                names = f["object_id"].asstr()[inside]
                locators = f["scene_id"].asstr()[inside]
                poses = f["object_pose_world"][inside]
                grasps = f["source_grasp_qpos"][inside]
            for j, name, locator, pose, q in zip(
                range(a, b), names, locators, poses, grasps, strict=True
            ):
                i = int(ids[j])
                assert parent["objects"][objects[i]] == name
                scale = frozen["groups"][int(geometries[i])]["scale"]
                mode = "tabletop" if tabletop[j] else "floating"
                # Source duplicates can be stored in different export packages.
                key = (mode, locator, tuple(scale), tuple(pose))
                if key not in scenes:
                    sid = len(scenes)
                    scenes[key] = sid
                    folder = out / "scenes"
                    folder.mkdir(exist_ok=True)
                    obj = dict(
                        type="rigid_object",
                        pose=pose.astype(float),
                        scale=np.array(scale),
                        **asset[name],
                    )
                    entities = {name: obj}
                    task = dict(obj_name=name, type="force_closure")
                    if mode == "tabletop":
                        entities["table"] = dict(
                            type="plane",
                            pose=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                            size=np.array([0.0, 0.0, 1.0]),
                        )
                        task.update(type="slide", axis=np.array([0.0, 0.0, 1.0]), distance=0.1)
                    np.save(
                        folder / f"{sid}.npy",
                        dict(
                            scene=entities,
                            scene_id=locator,
                            task=task,
                            pipeline_stage=(
                                "grasp_stability" if mode == "floating" else "tabletop_slide"
                            ),
                        ),
                    )
                sid = scenes[key]
                records[sid].append(
                    dict(index=i, object=int(objects[i]), grasp=q.tolist(), mode=mode)
                )
        offset = end
    assert sum(map(len, records.values())) == len(ids)
    save(
        out / "plan.json",
        dict(
            stages=str(stages),
            groups=frozen["models"],
            sources=records,
            hand=str(args.hand),
            spec=frozen["hand_spec"],
            selected=digest(root / "test_rows.npy" if args.all_rows else stages / "paired.npy"),
            selection="all" if args.all_rows else "ten_per_object",
            resident_source=digest(repo / "dexgrasp/test/resident_control.py"),
            cpu_adapter=digest(repo / "dexgrasp/test/dexonomy_control.py"),
            hand_sha=digest(args.hand),
            poses=frozen["pose_files"],
            code=digest(__file__),
            audit=digest(args.audit / "row_audit.npz"),
            task="Original Floating or Tabletop Slide then FC. Native refinement host, scene-specific QP wrench and mesh COM. Single-frame reconstructed control.",
            objects=parent["objects"],
            positive_baseline="archived successful GT in every selected row",
        ),
    )


def init_controls(out, plan):
    from omegaconf import OmegaConf
    from dexonomy.op.gen_grasp import GraspFilter

    global settings, repo_out, source_plan, arrays, frame
    repo_out = out
    source_plan = plan
    op = OmegaConf.load(repo / "simulation/dexonomy/config/op/grasp.yaml")
    settings = OmegaConf.create(
        dict(hand=dict(xml_path=plan["hand"]), op=OmegaConf.to_container(op, resolve=False))
    )
    arrays = {
        g: np.load(Path(plan["stages"]) / (g + ".npy"), mmap_mode="r") for g in plan["groups"]
    }
    frame = np.array(load(plan["spec"])["official_from_grasp_rotation"])


def build_commands(sid):
    from dexgrasp.test.dexonomy_control import controls
    from dexonomy.data.obj_loader import ObjSampleDataset
    from dexonomy.op.gen_grasp_warp_resident import _make_refinement_host
    from scipy.spatial.transform import Rotation

    dest = repo_out / "commands" / f"{sid}.npy"
    if dest.exists():
        return
    scene = str(repo_out / "scenes" / f"{sid}.npy")
    obs = ObjSampleDataset(1, 1, scene, 1, 0.1)[0]
    opt = _make_refinement_host(settings, scene, mixed=True)
    assert opt._cfg.physics_profile == "native"
    raw = obs["scene_cfg"]
    pose = raw["scene"][raw["task"]["obj_name"]]["pose"]
    world = Rotation.from_quat(pose[3:], scalar_first=True).as_matrix()
    names = [
        opt._model.joint(j).name.removeprefix(opt._cfg.hand_prefix)
        for j in range(opt._model.njnt)
        if 7 <= opt._model.jnt_qposadr[j] < 29
    ]
    spec = load(source_plan["spec"])
    assert sorted(names) == sorted(spec["joint_order"]) and len(names) == 22
    order = [spec["joint_order"].index(k) for k in names]
    result = []
    for row in source_plan["sources"][sid]:
        i = row["index"]
        for g, arr in arrays.items():
            p = arr[i]
            name = f"{g}/{i}"
            if not p["decoded"]:
                result.append(dict(id=name, invalid=True))
                continue
            quat = Rotation.from_matrix(world @ p["R"] @ frame.T).as_quat(scalar_first=True)
            q = np.r_[world @ p["t"] + pose[:3], quat, p["q"][order]]
            if g == "GT":
                source = np.asarray(row["grasp"], dtype=np.float32)
                np.testing.assert_allclose(q[:3], source[:3], atol=1e-6)
                np.testing.assert_allclose(q[7:], source[7:], atol=1e-6)
                assert abs(q[3:7] @ source[3:7]) > 1 - 1e-6
                q = source
            pre, squeeze, method = controls(
                opt, q, settings.op, qp_target=(obs["ext_wrench"], obs["ext_center"])
            )
            result.append(
                dict(
                    id=name,
                    control=method,
                    data=dict(
                        pipeline_stage="stability_candidate",
                        scene_path=scene,
                        pregrasp_qpos=pre,
                        grasp_qpos=q[None],
                        squeeze_qpos=squeeze,
                    ),
                )
            )
    with dest.with_suffix(".partial").open("wb") as f:
        np.save(f, result)
    dest.with_suffix(".partial").replace(dest)


def prepare_commands(args, plan):
    (args.output / "commands").mkdir(exist_ok=True)
    for g, proof in plan["poses"].items():
        if digest(Path(plan["stages"]) / (g + ".npy")) != proof:
            raise ValueError("Frozen pose file changed")
    with ProcessPoolExecutor(
        args.workers, initializer=init_controls, initargs=(args.output, plan)
    ) as pool:
        for i, _ in enumerate(pool.map(build_commands, plan["sources"], chunksize=1), 1):
            if i % 100 == 0:
                print("Prepared", i, "scenes", flush=True)


def simulate(args, plan):
    from dexonomy.op.eval_force_closure_warp_mixed import evaluate_force_closure_data_warp_mixed
    from dexonomy.tabletop_protocol import evaluate, make_host
    from dexonomy.op.gen_grasp_warp_resident import _make_refinement_host
    from dexonomy.data.obj_loader import ObjSampleDataset
    from dexgrasp.test.resident_control import resident_commands
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        dict(
            hand=dict(xml_path=plan["hand"]),
            obj_mass=0.1,
            physics_profile="native",
            warp_device="cuda:0",
            op=dict(
                miu_coef=[0.6, 0.02],
                hard=False,
                noslip_iterations=0,
                skip_pregrasp=False,
                trans_thre=0.05,
                rot_thre=15,
            ),
        )
    )
    refinement = OmegaConf.create(
        dict(
            hand=cfg.hand,
            warp_device="cuda:0",
            physics_profile="native",
            op=OmegaConf.to_container(
                OmegaConf.load(repo / "simulation/dexonomy/config/op/grasp.yaml"), resolve=False
            ),
        )
    )
    refinement.op.grasp.step = 0
    provenance = args.output / "controller"
    provenance.mkdir(exist_ok=True)
    out = args.output / "verdicts"
    out.mkdir(exist_ok=True)
    for sid in list(plan["sources"])[args.rank :: args.world_size]:
        target = out / f"{sid}.json"
        if target.exists():
            continue
        entries = np.load(args.output / "commands" / f"{sid}.npy", allow_pickle=True).tolist()
        valid = [x for x in entries if not x.get("invalid")]
        passed = {x["id"]: False for x in entries if x.get("invalid")}
        mode = plan["sources"][sid][0]["mode"]
        controls = {}
        if valid:
            scene = valid[0]["data"]["scene_path"]
            sample = ObjSampleDataset(1, 1, scene, 1, 0.1)[0]
            opt = _make_refinement_host(refinement, scene, mixed=True)
            for start in range(0, len(valid), 24):
                controls.update(
                    resident_commands(
                        opt,
                        valid[start : start + 24],
                        refinement,
                        (sample["ext_wrench"], sample["ext_center"]),
                    )
                )
        save(provenance / f"{sid}.json", controls)
        if valid and mode == "tabletop":
            host = make_host(valid[0]["data"]["scene_path"], plan["hand"])
            result = evaluate(
                valid, [host], [0] * len(valid), backend="warp", batch_size=min(30, len(valid) * 6)
            )
            passed.update((r["candidate_id"], bool(r["passes"])) for r in result["combined"])
        elif valid:
            result = evaluate_force_closure_data_warp_mixed(
                [x["data"] for x in valid],
                cfg,
                source_names=[x["id"] for x in valid],
                world_batch_size=min(90, len(valid) * 6),
            )
            if any(
                r.get("simulation_error") or r.get("metrics", {}).get("simulation_error")
                for r in result["results"]
            ):
                raise RuntimeError(f"Simulation error in scene {sid}")
            passed.update((r["source_name"], bool(r["success"])) for r in result["results"])
        assert len(passed) == len(entries)
        save(target, passed)


def summarize(args, plan):
    items = defaultdict(list)
    for sid, rows in plan["sources"].items():
        verdict = load(args.output / "verdicts" / f"{sid}.json")
        for row in rows:
            for group in plan["groups"]:
                v = verdict[f"{group}/{row['index']}"]
                items[group, "all"].append((row["object"], v))
                items[group, row["mode"]].append((row["object"], v))
    report = {}
    for (group, mode), pairs in items.items():
        successful = {o for o, v in pairs if v}
        total = len({o for o, v in pairs})
        n = len(pairs)
        wins = sum(v for o, v in pairs)
        report.setdefault(group, {})[mode] = dict(
            n=n,
            positive=wins,
            gsr=wins / n,
            objects=total,
            successful_objects=len(successful),
            osr=len(successful) / total,
        )
    save(args.output / "summary.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "controls", "run", "summarize"])
    p.add_argument("--output", required=True, type=Path)
    for k in ("root", "stages", "packages", "audit", "hand", "image-packages"):
        p.add_argument("--" + k, type=Path)
    p.add_argument("--all-rows", action="store_true")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    args = p.parse_args()
    for k in ("output", "root", "stages", "packages", "audit", "hand"):
        if getattr(args, k):
            setattr(args, k, getattr(args, k).resolve())
    if args.image_packages:
        sys.path.append(str(args.image_packages))
    import mujoco

    if mujoco.__version__ != "3.17.0":
        raise ValueError("This native source profile requires MuJoCo 3.17.0")
    if args.action == "prepare":
        prepare(args)
    else:
        plan = load(args.output / "plan.json")
        if (
            digest(__file__) != plan["code"]
            or digest(plan["hand"]) != plan["hand_sha"]
            or digest(repo / "dexgrasp/test/resident_control.py") != plan["resident_source"]
            or digest(repo / "dexgrasp/test/dexonomy_control.py") != plan["cpu_adapter"]
        ):
            raise ValueError("Source task implementation or hand differs from frozen plan")
        if args.action == "controls":
            prepare_commands(args, plan)
        elif args.action == "run":
            simulate(args, plan)
        else:
            summarize(args, plan)
