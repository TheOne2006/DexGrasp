"""Sampling, descriptive checks, and explicit adapters to IK/physical testing."""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import PHASES, batch, digest, load_manifest, normalize_split, write_json
from .methods import sample
from .models import GraspModel
from .train import seed_all


def generate(checkpoint, manifest_path, output, steps=None, candidates=None, device="cpu", split="test"):
    split = normalize_split(split)
    if split not in ("val", "test"):
        raise ValueError("sampling split must be val/test")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = state["config"]
    if digest(manifest_path) != state["manifest_sha256"]:
        raise ValueError("sampling manifest differs from training manifest")
    manifest = load_manifest(manifest_path)
    records = [r for r in manifest["records"] if r["split"] == split]
    if not records:
        raise ValueError("empty sampling split")
    steps = cfg["sample_steps"] if steps is None else steps
    candidates = cfg["candidates"] if candidates is None else candidates
    if type(steps) is not int or not 1 <= steps <= 1000 or type(candidates) is not int or candidates < 1:
        raise ValueError("invalid sampling steps/candidates")
    model = GraspModel(cfg["architecture"], cfg["width"], cfg["depth"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    seed_all(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    output.mkdir(parents=True)
    summary = {"checkpoint_sha256": digest(checkpoint), "manifest_sha256": digest(manifest_path),
               "split": split, "objective": cfg["objective"], "steps": steps,
               "nfe_per_candidate": steps, "candidates": candidates, "ranking": "none", "records": []}
    for i, row in enumerate(records):
        pc, _, centre = batch(records, [i], rng, cfg["num_points"], cfg["scale_m"], device, targets=False)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = sample(model, pc, cfg["objective"], steps, candidates)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        keypoints = result[0].cpu().numpy() * cfg["scale_m"] + centre[0]
        name = f"{i:06d}.npz"
        np.savez_compressed(output / name, keypoints=keypoints,
                            point_cloud=pc[0].cpu().numpy() * cfg["scale_m"] + centre[0])
        summary["records"].append({"file": name, "object_id": row["object_id"], "id": row["id"],
                                   "scene_path": row.get("scene_path"), "generation_seconds": seconds})
    write_json(output / "samples.json", summary)
    return summary


def metrics(folder):
    folder = Path(folder)
    meta = json.loads((folder / "samples.json").read_text())
    if not meta["records"]:
        raise ValueError("empty samples")
    spread, seconds, count = [], [], 0
    for row in meta["records"]:
        with np.load(folder / row["file"], allow_pickle=False) as f:
            kp = f["keypoints"]
        if kp.ndim != 4 or kp.shape[1:] != (3, 28, 3) or not len(kp) or not np.isfinite(kp).all():
            raise ValueError("invalid generated keypoints")
        spread.append(float(kp.std(axis=0).mean() * 1000))
        seconds.append(row["generation_seconds"])
        count += len(kp)
    return {"status": "descriptive_only", "records": len(spread), "grasps": count,
            "mean_coordinate_std_mm": float(np.mean(spread)),
            "mean_generation_seconds_per_record": float(np.mean(seconds)),
            "physical_success_rate": None, "penetration": None,
            "note": "Coordinate spread is not grasp coverage or success; latency includes encoding, excludes IK/simulation, no warmup."}


def reconstruct(folder, output, kpgrasp_root):
    """Reuse upstream solver without changing its checkout or output guards."""
    folder, output, root = Path(folder), Path(output), Path(kpgrasp_root)
    meta = json.loads((folder / "samples.json").read_text())
    if output.exists():
        raise FileExistsError(output)
    if not meta["records"] or any(not r.get("scene_path") or not Path(r["scene_path"]).is_file()
                                  for r in meta["records"]):
        raise ValueError("physical export requires a valid scene_path for every record")
    module_file = root / "src/dexgrasp/kinematics/ik.py"
    spec = importlib.util.spec_from_file_location("kpgrasp_ik", module_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    solver = module.ShadowHandIK(root / "assets/hand/shadow/right_hand_w_kp.xml")
    output.mkdir(parents=True)
    stats = []
    for i, row in enumerate(meta["records"]):
        with np.load(folder / row["file"], allow_pickle=False) as f:
            keypoints = f["keypoints"]
        for j, kp in enumerate(keypoints):
            record = {"scene_path": row["scene_path"]}
            for phase, points in zip(PHASES, kp):
                qpos, info = solver.solve(points, return_stats=True)
                if qpos.shape != (29,) or not np.isfinite(qpos).all():
                    raise ValueError("invalid Shadow Hand pose/joints")
                record[f"{phase}_qpos"] = qpos
                record[f"{phase}_kp"] = points
                stats.append({"record": row["id"], "candidate": j, "phase": phase, **info})
            np.save(output / f"{i:06d}_{j:04d}.npy", record)
    report = {"status": "ik_only", "physical_success_rate": None, "phases": stats,
              "solver_sha256": digest(module_file), "samples_sha256": digest(folder / "samples.json")}
    write_json(output / "ik_stats.json", report)
    return report


def benchmark_plan(root, python, samples, output, name):
    """Three isolated stages; no upstream shell script with rm -r is invoked."""
    common = [python, "src/main.py", "hand=shadow", f"exp_name={name}",
              f"save_root={Path(output).resolve()}", "n_worker=4", "setting=fc"]
    return {"status": "planned", "cwd": str(Path(root).resolve()),
            "protocol": {"branch": "main", "mass_kg": 0.1, "friction": [0.6, 0.02],
                         "max_penetration_m": 0.01, "translation_threshold_m": 0.05,
                         "angle_threshold_deg": 15},
            "commands": [common + ["task=format", "task.data_name=Learning", "task.max_num=-1",
                                     f"task.data_path={Path(samples).resolve()}"],
                         common + ["task=eval", "task.max_num=-1", "task.obj_mass=0.1",
                                   "task.miu_coef=[0.6,0.02]", "task.simulation_metrics.max_pene=0.01",
                                   "task.simulation_metrics.trans_thre=0.05", "task.simulation_metrics.angle_thre=15"],
                         common + ["task=stat"]]}
