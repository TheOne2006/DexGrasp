"""Use the external Dexonomy checkout in its own interpreter to test decoded poses."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dexgrasp.test import dexonomy_control
from dexgrasp.test.dexonomy_control import make_scene, test_pose
from dexgrasp.test.frozen import decoded, digest as sha, save


def read(path):
    return json.loads(path.read_text())


def evaluate_scene(index):
    from dexonomy.util.file_util import load_scene_cfg

    scene = load_scene_cfg(plan["scenes"][index]["path"])
    context = make_scene(scene, hand_xml, spec, settings)
    from dexgrasp.test.bench_force import build_replay

    obj = scene["scene"][scene["task"]["obj_name"]]
    np.testing.assert_allclose(obj["scale"], obj["scale"][0])
    bench = build_replay(
        bench_backend,
        Path(obj["info_path"]).parents[1],
        obj["scale"][0],
        obj["density"],
        hand_xml,
        bench_config,
        obj["pose"],
    )
    result = {}
    for g, poses in arrays.items():
        values = []
        for p in poses[index * 10 : index * 10 + 10]:
            if p["decoded"]:
                outcomes, source = test_pose(context, p, frame, settings, bench=bench)
                values.append([outcomes["derived"], source, outcomes["bench"]])
            else:
                values.append([False, "decode_failed", False])
        result[g] = values
    return index, result


def main(args):
    if args.image_packages:
        sys.path.append(str(args.image_packages.resolve()))
    from omegaconf import OmegaConf
    from dexgrasp.test.bench_force import load_backend, load_config

    global plan, settings, spec, hand_xml, arrays, frame
    root = args.inputs
    sys.path.insert(0, str(args.reference.resolve()))
    from dexonomy.sim import MuJoCo_EvalEnv
    from dexonomy.op.gen_grasp import GraspFilter  # Load Torch before forking workers.

    global bench_backend, bench_config
    bench_backend = load_backend(args.bench)
    bench_config = load_config(args.bench)
    import inspect

    assert Path(inspect.getfile(MuJoCo_EvalEnv)).resolve().is_relative_to(args.reference.resolve())
    assert mujoco.__version__ == "3.3.3"
    plan = read(root / "plan.json")
    if args.assets_root:
        old = Path(plan["assets_root"])
        for scene in plan["scenes"]:
            scene["path"] = str(args.assets_root.resolve() / Path(scene["path"]).relative_to(old))
    settings = OmegaConf.load(args.reference / "dexonomy/config/op/grasp.yaml")
    spec = read(Path(plan["hand_spec"]))
    frame = np.asarray(spec["official_from_grasp_rotation"])
    hand_xml = args.hand.resolve()
    arrays = {}
    checks = {}
    for g in "ABCD":
        paths = [root / g / f"rank{rank}" / "official/decoded.npy" for rank in (0, 1)]
        a = decoded(root, g, "official")
        assert len(a) == 10 * len(plan["scenes"])
        arrays[g] = a
        checks[g] = [sha(p) for p in paths]
    output = root / "physics"
    output.mkdir(exist_ok=True)
    chunks = output / "scenes"
    chunks.mkdir(exist_ok=True)
    protocol = dict(
        plan=sha(root / "plan.json"),
        poses=checks,
        source=sha(Path(__file__)),
        control_source=sha(dexonomy_control.__file__),
        upstream=subprocess.check_output(
            ["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True
        ).strip(),
        bench_upstream=subprocess.check_output(
            ["git", "-C", str(args.bench), "rev-parse", "HEAD"], text=True
        ).strip(),
        bench_adapter=sha(Path(__file__).resolve().parents[1] / "dexgrasp/test/bench_force.py"),
        hand=sha(hand_xml),
        mujoco=mujoco.__version__,
        controls="Dexonomy QP squeeze and pregrasp margin rollout; hold grasp target if contact QP unavailable; all candidates counted",
        force="upstream eval_fc, mass 0.1 kg, friction 0.6/0.02, hard=false, 5 cm, 15 deg",
    )
    if (output / "protocol.json").exists() and read(output / "protocol.json") != protocol:
        raise ValueError("Inputs changed since previous evaluation")
    save(output / "protocol.json", protocol)
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [
            pool.submit(evaluate_scene, i)
            for i in range(len(plan["scenes"]))
            if not (chunks / f"{i}.json").exists()
        ]
        for done, future in enumerate(as_completed(futures), 1):
            i, rows = future.result()
            save(chunks / f"{i}.json", rows)
            if done % 100 == 0:
                print(done, "/", len(futures), flush=True)
    from dexgrasp.test.force_summary import summarize_official

    save(output / "summary.json", summarize_official(root, output))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument(
        "--bench",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "reference/DexGraspBench",
    )
    p.add_argument(
        "--assets-root",
        type=Path,
        help="Relocate frozen scene files to an identical official asset tree",
    )
    p.add_argument("--hand", type=Path, required=True)
    p.add_argument(
        "--image-packages",
        type=Path,
        help="Optional training-image packages for upstream Torch imports; benchmark packages take precedence",
    )
    p.add_argument("--workers", type=int, default=64)
    main(p.parse_args())
