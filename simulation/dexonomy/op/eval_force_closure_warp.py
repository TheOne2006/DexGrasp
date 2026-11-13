"""Shared six-direction helpers for the V1.3.1 mixed-world evaluator."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np

from dexonomy.sim import HandCfg, MuJoCo_EvalCfg, MuJoCo_EvalEnv
from dexonomy.sim.mujoco_env import physics_stage_arena_memory_bytes
from dexonomy.util.file_util import load_json, load_scene_cfg
from dexonomy.util.np_util import np_get_delta_pose, np_interp_qpos, np_interp_slide
from dexonomy.util.traj_util import get_planner


FORCE_DIRECTIONS = np.asarray(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=np.float32,
)


class MuJoCoWarpEvaluationError(RuntimeError):
    """Raised when an input or a Warp simulation cannot be evaluated safely."""



def _prepare_scene(scene_path: str, obj_mass: float) -> dict[str, Any]:
    scene = load_scene_cfg(scene_path)
    target_name = scene.get("task", {}).get("obj_name")
    if (
        scene.get("pipeline_stage") != "grasp_stability"
        or scene.get("task", {}).get("type") != "force_closure"
        or set(scene.get("scene", {})) != {target_name}
        or scene["scene"][target_name].get("fix_root", False)
    ):
        raise MuJoCoWarpEvaluationError(
            f"expected an isolated movable force-closure target: {scene_path}"
        )
    obj_cfg = scene["scene"][target_name]
    info = load_json(obj_cfg["info_path"])
    mass_coefficient = info["mass"] / (info["density"] * info["scale"] ** 3)
    obj_cfg["density"] = obj_mass / (
        mass_coefficient * float(np.prod(obj_cfg["scale"]))
    )
    return scene



def _make_environment(
    scene: dict[str, Any],
    hand_xml: str,
    miu_coef: tuple[float, float],
    hard: bool,
    noslip_iterations: int = 2,
    physics_profile: str = "upstream_mujoco333",
) -> MuJoCo_EvalEnv:
    if noslip_iterations < 0:
        raise ValueError("noslip_iterations must be non-negative")
    env = MuJoCo_EvalEnv(
        hand_cfg=HandCfg(hand_xml, freejoint=True),
        scene_cfg=scene,
        sim_cfg=MuJoCo_EvalCfg(
            miu_coef=list(miu_coef), noslip_iterations=noslip_iterations,
            arena_memory_bytes=physics_stage_arena_memory_bytes(),
            physics_profile=physics_profile,
        ),
        debug_render=False,
        debug_view=False,
        hard=hard,
    )
    return env



def _check_noslip_support(model, warp_model) -> None:
    """Reject backends that cannot preserve the requested noslip solver."""
    if model.opt.noslip_iterations == 0:
        return
    if (
        not hasattr(warp_model.opt, "noslip_iterations")
        or not hasattr(warp_model.opt, "noslip_tolerance")
        or int(warp_model.opt.noslip_iterations) != model.opt.noslip_iterations
    ):
        raise MuJoCoWarpEvaluationError(
            "MuJoCo Warp did not preserve the requested noslip options; "
            "install a noslip-capable version (3.17.0 or newer), or explicitly "
            "set noslip_iterations=0 to reproduce the legacy configuration"
        )



def _candidate_trajectory(
    env: MuJoCo_EvalEnv,
    scene: dict[str, Any],
    candidate: dict[str, Any],
    skip_pregrasp: bool,
) -> tuple[list[np.ndarray], list[str], np.ndarray]:
    if skip_pregrasp:
        candidate = dict(candidate)
        candidate["pregrasp_qpos"] = candidate["grasp_qpos"].copy()
    planner = get_planner(scene["task"])
    ctrl_qpos, ctrl_type, _, target_pose = planner.plan_trajectory(
        np.copy(env.get_active_obj_pose()),
        candidate["pregrasp_qpos"],
        candidate["grasp_qpos"],
        candidate["squeeze_qpos"],
        candidate.get("robot_pose"),
    )
    return [np.asarray(value) for value in ctrl_qpos], ctrl_type, target_pose



def _controls_for_worlds(
    env: MuJoCo_EvalEnv, hand_qpos: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert per-candidate hand targets to actuator and mocap values."""
    ctrl = np.stack(
        [env._hand_qpos2ctrl_mat[:, 6:] @ qpos[7:] for qpos in hand_qpos], axis=0
    ).astype(np.float32)
    mocap_pos = hand_qpos[:, None, :3].astype(np.float32)
    mocap_quat = hand_qpos[:, None, 3:7].astype(np.float32)
    return ctrl, mocap_pos, mocap_quat



def _interpolate_segment(
    start: np.ndarray, stop: np.ndarray, ctrl_type: str, steps: int
) -> np.ndarray:
    result = np_interp_qpos(start, stop, steps)
    if ctrl_type == "ee_pose":
        result[:, :7] = np_interp_slide(start[:7], stop[:7], steps)
    elif ctrl_type != "joint_angle":
        raise MuJoCoWarpEvaluationError(f"unsupported control type: {ctrl_type}")
    return result



def _metrics_from_checkpoints(
    target_pose: np.ndarray,
    checkpoint_poses: list[np.ndarray],
    trans_threshold: float,
    rot_threshold: float,
) -> list[dict[str, Any]]:
    if not np.isfinite(target_pose).all():
        raise MuJoCoWarpEvaluationError("non-finite target pose during stability evaluation")
    nworld = checkpoint_poses[0].shape[0]
    trans = np.zeros((nworld, len(checkpoint_poses)), dtype=np.float64)
    rot = np.zeros_like(trans)
    for hold_index, poses in enumerate(checkpoint_poses):
        if not np.isfinite(poses).all():
            raise MuJoCoWarpEvaluationError(
                f"non-finite checkpoint pose at hold {hold_index + 1}"
            )
        for world_index, pose in enumerate(poses):
            trans[world_index, hold_index], rot[world_index, hold_index] = (
                np_get_delta_pose(target_pose, pose)
            )
    if not np.isfinite(trans).all() or not np.isfinite(rot).all():
        raise MuJoCoWarpEvaluationError("non-finite stability pose error")

    candidate_count = nworld // len(FORCE_DIRECTIONS)
    results = []
    for candidate_index in range(candidate_count):
        direction_results = []
        passes_all = True
        for direction_index, direction in enumerate(FORCE_DIRECTIONS):
            world_index = candidate_index * len(FORCE_DIRECTIONS) + direction_index
            failed = np.flatnonzero(
                (trans[world_index] >= trans_threshold)
                | (rot[world_index] >= rot_threshold)
            )
            completed = int(failed[0] + 1) if len(failed) else len(checkpoint_poses)
            direction_passes = len(failed) == 0
            direction_results.append(
                {
                    "direction": direction.astype(float).tolist(),
                    "max_translation_error_m": float(
                        np.max(trans[world_index, :completed])
                    ),
                    "max_rotation_error_deg": float(
                        np.max(rot[world_index, :completed])
                    ),
                    "completed_holds": completed,
                    "passes": direction_passes,
                }
            )
            if not direction_passes:
                passes_all = False
                # Preserve all six measured directions in the prediction report.
        worst_trans = max(
            item["max_translation_error_m"] for item in direction_results
        )
        worst_rot = max(item["max_rotation_error_deg"] for item in direction_results)
        results.append(
            {
                "type": "force_closure_6d",
                "backend": "mujoco_warp",
                "passes": bool(
                    passes_all and len(direction_results) == len(FORCE_DIRECTIONS)
                ),
                "directions_completed": len(direction_results),
                "directions_required": len(FORCE_DIRECTIONS),
                "worst_translation_error_m": worst_trans,
                "worst_rotation_error_deg": worst_rot,
                "translation_threshold_m": float(trans_threshold),
                "rotation_threshold_deg": float(rot_threshold),
                "stability_score": max(
                    worst_trans / trans_threshold, worst_rot / rot_threshold
                ),
                "direction_results": direction_results,
            }
        )
    return results
