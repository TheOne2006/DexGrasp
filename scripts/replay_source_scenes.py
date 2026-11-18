"""Stream prepared source scenes through native MuJoCo Warp with persistent Floating model banks."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

repo = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(repo / "simulation"), str(repo)]
from dexgrasp.test.frozen import digest, save


def replay(entries, mode, plan, cfg, refinement, limits):
    from dexgrasp.test.resident_control import resident_commands
    from dexonomy.data.obj_loader import ObjSampleDataset
    from dexonomy.op.eval_force_closure_warp_mixed import MixedForceClosureBatcher
    from dexonomy.op.gen_grasp_warp_resident import _make_refinement_host
    from dexonomy.tabletop_protocol import evaluate, make_host

    valid = [r for r in entries if not r.get("invalid")]
    verdicts = {r["id"]: False for r in entries if r.get("invalid")}
    controllers = {}
    if not valid:
        return verdicts, controllers
    scene = valid[0]["data"]["scene_path"]
    sample = ObjSampleDataset(1, 1, scene, 1, 0.1)[0]
    target = sample["ext_wrench"], sample["ext_center"]
    opt = _make_refinement_host(refinement, scene, mixed=True)
    for start in range(0, len(valid), limits["refinement"]):
        part = valid[start : start + limits["refinement"]]
        controllers.update(resident_commands(opt, part, refinement, target))

    if mode == "tabletop":
        host = make_host(scene, plan["hand"])
        result = evaluate(
            valid,
            [host],
            [0] * len(valid),
            backend="warp",
            batch_size=min(limits["slide_force"], 6 * len(valid)),
        )
        if len(result["combined"]) != len(valid):
            raise RuntimeError(f"Incomplete Slide+FC results for {scene}")
        verdicts.update((row["candidate_id"], bool(row["passes"])) for row in result["combined"])
    else:
        bank = MixedForceClosureBatcher(
            cfg, min(limits["floating_worlds"], len(valid) * 6), scene_paths=[scene]
        )
        first = bank.push([r["data"] for r in valid], source_names=[r["id"] for r in valid])
        last = bank.flush()
        for item in first["results"] + last["results"]:
            metrics = item.get("metrics") or {}
            if (
                metrics.get("simulation_error")
                or metrics.get("type") == "simulation_error"
                or "error" in metrics
            ):
                raise RuntimeError(f"Invalid physical evaluation: {item['source_name']}")
            verdicts[item["source_name"]] = bool(item["success"])
    if set(verdicts) != {r["id"] for r in entries}:
        raise RuntimeError("Candidate identity mismatch")
    return verdicts, controllers


def main(args):
    from importlib.metadata import version
    from omegaconf import OmegaConf

    source = args.inputs.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    verdicts = output / "verdicts"
    provenance = output / "controller"
    verdicts.mkdir(exist_ok=True)
    provenance.mkdir(exist_ok=True)
    plan = json.loads((source / "plan.json").read_text())
    controls = repo / "dexgrasp/test/resident_control.py"
    if digest(controls) != plan["resident_source"] or digest(plan["hand"]) != plan["hand_sha"]:
        raise ValueError("Inputs or resident controller have changed")
    if version("mujoco") != "3.17.0" or version("warp-lang") != "1.16.0":
        raise ValueError("Source replay requires the production solver versions")
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
            warp_device=cfg.warp_device,
            physics_profile="native",
            op=OmegaConf.to_container(
                OmegaConf.load(repo / "simulation/dexonomy/config/op/grasp.yaml"), resolve=False
            ),
        )
    )
    refinement.op.grasp.step = 0
    os.environ["DEXONOMY_DATA_REUSE"] = "1"
    limits = dict(
        refinement=args.refinement,
        floating_worlds=args.floating_worlds,
        slide_force=args.slide_force,
    )
    cfg_record = dict(
        parent=digest(source / "plan.json"),
        implementation=digest(__file__),
        resident_source=digest(controls),
        upstream=digest(repo / "simulation/dexonomy/op/eval_force_closure_warp_mixed.py"),
        mujoco=version("mujoco"),
        mjwarp=version("mujoco-warp"),
        warp=version("warp-lang"),
        limits=limits,
        floating_graph_reuse=True,
        selection=args.scenes,
    )
    receipt = output / f"protocol-{args.rank}.json"
    if receipt.exists() and json.loads(receipt.read_text()) != cfg_record:
        raise ValueError("Resume protocol mismatch")
    save(receipt, cfg_record)
    keys = list(plan["sources"]) if args.scenes is None else [str(i) for i in args.scenes]
    pending = {
        k for k in keys[args.rank :: args.world_size] if not (verdicts / f"{k}.json").exists()
    }
    while pending:
        available = sorted(pending & {p.stem for p in (source / "commands").glob("*.npy")}, key=int)
        if not available:
            time.sleep(10)
            continue
        for sid in available:
            entries = np.load(source / "commands" / f"{sid}.npy", allow_pickle=True).tolist()
            mode = plan["sources"][sid][0]["mode"]
            values, method = replay(entries, mode, plan, cfg, refinement, limits)
            save(provenance / f"{sid}.json", method)
            save(verdicts / f"{sid}.json", values)
            pending.remove(sid)
            print(sid, len(values), len(pending), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--scenes", type=int, nargs="+")
    p.add_argument("--refinement", type=int, default=96)
    p.add_argument("--floating-worlds", type=int, default=640)
    p.add_argument("--slide-force", type=int, default=192)
    p.add_argument("--image-packages", type=Path)
    args = p.parse_args()
    if args.image_packages:
        sys.path.append(str(args.image_packages))
    if (
        not 0 <= args.rank < args.world_size
        or min(args.refinement, args.floating_worlds, args.slide_force) < 1
    ):
        p.error("Invalid batch size or rank")
    main(args)
