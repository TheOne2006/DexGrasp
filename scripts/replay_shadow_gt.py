"""Paired released Shadow control ablations in Dexonomy and the KPGrasp evaluator."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np


def run(row):
    from dexonomy.sim import HandCfg, MuJoCo_EvalCfg, MuJoCo_EvalEnv
    from dexonomy.util.file_util import load_scene_cfg
    from dexonomy.util.traj_util import get_planner
    from dexgrasp.test.bench_force import build_replay, evaluate
    from dexgrasp.data.dexonomy_scale import released_grasp_scale

    archive = np.load(row["grasp"], allow_pickle=True).item()
    i = row["index"]
    # The released collection augments each candidate with its own scale factor.
    # DexGraspBench/src/task/convert_format.py::Batched and KPGrasp/data_proc.py.
    scene_path = Path(row["scene"])
    scene = load_scene_cfg(str(scene_path))
    obj = scene["scene"][scene["task"]["obj_name"]]
    base_scale = np.asarray(obj["scale"]).copy()
    factor = float(archive["scene_scale"][i])
    obj["scale"] = released_grasp_scale(base_scale, factor)
    np.testing.assert_allclose(obj["scale"], obj["scale"][0])
    info = json.loads(Path(obj["info_path"]).read_text())
    obj["density"] = 0.1 / (
        info["mass"] / (info["density"] * info["scale"] ** 3) * np.prod(obj["scale"])
    )
    env = MuJoCo_EvalEnv(
        HandCfg(str(dex_hand), freejoint=True),
        scene,
        MuJoCo_EvalCfg(miu_coef=[0.6, 0.02]),
        debug_view=False,
    )
    bench = build_replay(
        backend,
        Path(obj["info_path"]).parents[1],
        obj["scale"][0],
        obj["density"],
        bench_hand,
        cfg,
        obj["pose"],
    )
    target = env.get_active_obj_pose().copy()
    pre, grasp, squeeze = [
        archive[key][i] for key in ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos")
    ]
    result = {key: row[key] for key in ("object", "scene", "grasp", "index")}
    result.update(
        scale_factor=factor,
        base_scale=float(base_scale[0]),
        actual_scale=float(obj["scale"][0]),
        dexonomy={},
        dexgraspbench={},
    )
    for name, stages in (
        ("released", (pre, grasp, squeeze)),
        ("no_squeeze", (pre, grasp, grasp)),
        ("hold", (grasp, grasp, grasp)),
    ):
        trajectory = get_planner(scene["task"]).plan_trajectory(
            target.copy(), *[s[None] for s in stages], None
        )
        passed, _, _ = env.eval_fc(*trajectory, 0.05, 15)
        result["dexonomy"][name] = bool(passed and not env._data.warning.number.any())
        result["dexgraspbench"][name] = evaluate(bench, *stages)
    return result


def main(args):
    global backend, cfg, dex_hand, bench_hand
    sys.path.insert(0, str(args.dexonomy.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if args.image_packages:
        sys.path.append(str(args.image_packages))
    from dexgrasp.test.bench_force import load_backend, load_config
    from dexgrasp.test.frozen import digest, save
    import mujoco

    bench_hand = args.bench.resolve() / "assets/hand/shadow/right_hand.xml"
    dex_hand = args.dexonomy.resolve() / "assets/hand/shadow/right.xml"
    backend = load_backend(args.bench)
    cfg = load_config(args.bench)
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.candidates.read_text())
    save(
        root / "protocol.json",
        dict(
            selected=digest(args.candidates),
            source=digest(__file__),
            upstream_converter=digest(args.bench / "src/task/convert_format.py"),
            dexonomy_hand=digest(dex_hand),
            bench_hand=digest(bench_hand),
            mujoco=mujoco.__version__,
            candidates=len(rows),
            scale="scene_cfg.scale multiplied by per-candidate scene_scale",
        ),
    )
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(run, rows))
    save(root / "candidates.json", results)
    summary = {}
    for engine in ("dexonomy", "dexgraspbench"):
        summary[engine] = {}
        for mode in ("released", "no_squeeze", "hold"):
            passed = [r for r in results if r[engine][mode]]
            summary[engine][mode] = dict(
                n=len(results),
                successes=len(passed),
                objects=len({r["object"] for r in results}),
                successful_objects=len({r["object"] for r in passed}),
                gsr=len(passed) / len(results),
            )
    save(root / "summary.json", summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("candidates", "output", "dexonomy", "bench"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--image-packages", type=Path)
    p.add_argument("--workers", type=int, default=24)
    main(p.parse_args())
