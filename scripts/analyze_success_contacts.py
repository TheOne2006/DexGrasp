"""Contact-set diversity at the beginning of the force test for successful noise draws."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np


def record_object(job):
    obj, ids = job
    from scipy.spatial.distance import pdist
    from scipy.spatial.transform import Rotation
    from dexonomy.util.traj_util import get_planner
    from dexgrasp.test.dexonomy_control import make_scene, controls, floating_scene
    import mujoco

    gid = group_by_observation[observations[ids[0]]]
    scene = floating_scene(stages / "assets", plan["groups"][gid])
    scene, opt, env, target, order = make_scene(scene, hand_xml, spec, settings)
    bodies = env._model.geom_bodyid
    names = [env._model.body(j).name for j in range(env._model.nbody)]
    hp = env._cfg.hand_prefix
    data = {}
    for j, (group, arr) in enumerate(poses.items()):
        patterns = []
        centers_on_object = []
        for i in ids[passed[j, ids]]:
            pose = arr[i]
            orientation = Rotation.from_matrix(pose["R"] @ frame.T).as_quat(scalar_first=True)
            grasp = np.r_[pose["t"] + centers[observations[i]], orientation, pose["q"][order]][None]
            pre, squeeze, _ = controls(opt, grasp[0], settings)
            ctrl, types, forces, _ = get_planner(scene["task"]).plan_trajectory(
                target.copy(), pre, grasp, squeeze, None
            )
            env.reset_qpos(ctrl[0])
            env.run_trajectory(ctrl, types, forces)
            mujoco.mj_forward(env._model, env._data)
            contacts_by_link = {}
            body = env._obj_active_bodyid
            obj_origin = env._data.xpos[body]
            obj_rotation = env._data.xmat[body].reshape(3, 3)
            for contact in env._data.contact:
                pair = [int(bodies[k]) for k in (contact.geom1, contact.geom2)]
                if body in pair and contact.dist <= 0.002:
                    for k in pair:
                        if names[k].startswith(hp):
                            point = (contact.pos - obj_origin) @ obj_rotation
                            contacts_by_link.setdefault(names[k].removeprefix(hp), []).append(point)
            patterns.append(tuple(sorted(contacts_by_link)))
            if contacts_by_link:
                centers_on_object.append(
                    np.mean(
                        [np.mean(points, axis=0) for points in contacts_by_link.values()], axis=0
                    )
                )
        if len(patterns) >= 2:
            counts = Counter(patterns)
            n = len(patterns)
            data[group] = dict(
                successes=n,
                distinct=len(counts),
                zero_contact=sum(not s for s in patterns),
                pair_disagreement=1 - sum(c * (c - 1) for c in counts.values()) / (n * (n - 1)),
                draws_with_contacts=len(centers_on_object),
                contact_centroid_separation_mm=(
                    float(pdist(centers_on_object).mean() * 1000)
                    if len(centers_on_object) >= 2
                    else None
                ),
                sets=[dict(links=list(k), count=v) for k, v in counts.items()],
            )
    return str(obj), data


def main(args):
    global stages, plan, settings, hand_xml, frame, spec, poses, observations, centers, group_by_observation, passed
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(args.reference.resolve()))
    if args.image_packages:
        sys.path.append(str(args.image_packages))
    from dexgrasp.test.frozen import decoded, save, digest
    from dexonomy.op.gen_grasp import GraspFilter
    from omegaconf import OmegaConf

    stages = args.stages.resolve()
    plan = json.loads((stages / "plan.json").read_text())
    settings = OmegaConf.load(args.reference / "dexonomy/config/op/grasp.yaml")
    hand_xml = args.hand.resolve()
    spec = json.loads(Path(plan["hand_spec"]).read_text())
    frame = np.array(spec["official_from_grasp_rotation"])
    poses = {g: decoded(args.inputs, g, "noise") for g in "ABCD"}
    passed = np.load(args.force / "outcomes.npy").astype(bool)
    n = passed.shape[1]
    rows, objects = (np.empty(n, dtype=int) for _ in range(2))
    for rank in (0, 1):
        with np.load(args.inputs / "A" / f"rank{rank}" / "noise/index.npz") as idx:
            rows[rank::2] = idx["packed"]
            objects[rank::2] = idx["objects"]
    parent = json.loads((args.root / "plan.json").read_text())
    observations = np.load(Path(parent["root"]) / "observation_index.npy", mmap_mode="r")[rows]
    centers = np.load(args.root / "quality/centers.npy")
    group_by_observation = {o: g["id"] for g in plan["groups"] for o in g["observations"]}
    jobs = [(parent["objects"][obj], np.flatnonzero(objects == obj)) for obj in np.unique(objects)]
    with ProcessPoolExecutor(args.workers) as pool:
        records = dict(pool.map(record_object, jobs))
    summary = {}
    for name in poses:
        rows = [obj[name] for obj in records.values() if name in obj]
        spreads = [
            r["contact_centroid_separation_mm"]
            for r in rows
            if r["contact_centroid_separation_mm"] is not None
        ]
        summary[name] = dict(
            objects_with_two_successes=len(rows),
            objects_with_distinct_sets=sum(r["distinct"] > 1 for r in rows),
            mean_pair_disagreement=(
                float(np.mean([r["pair_disagreement"] for r in rows])) if rows else None
            ),
            empty_sets=sum(r["zero_contact"] for r in rows),
            objects_with_contact_centroid_pairs=len(spreads),
            mean_contact_centroid_separation_mm=float(np.mean(spreads)) if spreads else None,
        )
    save(
        args.force / "contact_diversity.json",
        dict(
            summary=summary,
            per_object=records,
            selection="Six-direction successes. Active hand-object contacts at start of force application, <=2mm.",
            centroid="Mean of per-link mean contact locations in the object's body frame. Mean distance across unordered pairs of draws with contacts, then equal weight per eligible object.",
            frozen_outcomes=digest(args.force / "outcomes.npy"),
            source=digest(__file__),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "stages", "inputs", "force", "reference", "hand"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--image-packages", type=Path)
    p.add_argument("--workers", type=int, default=32)
    main(p.parse_args())
