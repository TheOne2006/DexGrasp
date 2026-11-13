"""V1.3 force closure: mixed-object worlds, including split direction sets.

The streaming batcher never pads a batch with extra attempts. A world is one
candidate and one external-force direction, and each candidate is returned only
after its six worlds have completed. The historical trajectory and acceptance
calculations are reused from the original evaluator.
"""

from __future__ import annotations
from dexonomy.acceptance_audit import event as audit_event, checkpoints as audit_checkpoints
from dexonomy.sim.fc_execution import direction_tasks, add_direction, ordered_directions, checked_steps, capture_graph, launch_graph, check_execution

from collections import deque
from dataclasses import dataclass, field
import os
import time
from typing import Any

import mujoco
import numpy as np

from .eval_force_closure_warp import (
    FORCE_DIRECTIONS,
    MuJoCoWarpEvaluationError,
    _candidate_trajectory,
    _check_noslip_support,
    _controls_for_worlds,
    _interpolate_segment,
    _make_environment,
    _metrics_from_checkpoints,
    _prepare_scene,
)


@dataclass
class _Candidate:
    data: dict[str, Any]
    source_name: str
    env: Any
    target_pose: np.ndarray
    initial_hand: np.ndarray
    control_qpos: np.ndarray
    checkpoint_poses: dict[int, np.ndarray] = field(default_factory=dict)
    checkpoint_seen: dict[int, np.ndarray] = field(default_factory=dict)
    terminal_state: dict[int, dict] = field(default_factory=dict)
    solver_flags: dict[int, int] = field(default_factory=dict)
    batch_ids: set[int] = field(default_factory=set)


def _round_controls(trajectory) -> np.ndarray:
    controls, control_types, _ = trajectory
    if len(controls) < 2:
        raise MuJoCoWarpEvaluationError("force-closure trajectory has no preamble")
    return np.concatenate([
        _interpolate_segment(start, stop, control_types[index], 5)
        for index, (start, stop) in enumerate(zip(controls[:-1], controls[1:]))
    ], axis=0)


def _release_batch_memory(device):
    from dexonomy.sim.mujoco_warp_env import release_warp_scene_memory
    release_warp_scene_memory(device)


def _simulate_world_batch(worlds, cfg, allocation_scale=1, model_bank=None):
    """Run independent per-world preambles and ten 50-step force holds."""
    import mujoco_warp as mjw
    import warp as wp

    from dexonomy.sim import warp_stability_kernels as k
    from dexonomy.sim.mujoco_warp_env import (
        MuJoCoWarpAllocationError,
        check_warp_allocation,
        preserve_model_options,
    )
    from dexonomy.sim.mujoco_warp_mixed import build_mixed_model
    from dexonomy.sim.mujoco_warp_resident_env import ResidentRefinementBatch
    from dexonomy.sim.warp_refinement_kernels import copy_world_array

    started = time.perf_counter()
    timings = {}
    device = str(getattr(cfg, "warp_device", "cuda:0"))
    wp.set_device(device)
    if mujoco.__version__.split(".")[:2] != ["3", "17"] or mjw.__version__.split(".")[:2] != ["3", "17"]:
        raise MuJoCoWarpEvaluationError("V1.3 requires MuJoCo and MuJoCo Warp 3.17")
    import os
    retirement = bool(cfg.get("eval_failed_world_retirement", False))
    reuse_graph = model_bank is not None and os.environ.get("DEXONOMY_DATA_REUSE") == "1"
    lengths = [len(candidate.control_qpos) for candidate, _ in worlds]
    graph_key = (len(worlds), max(lengths), len(set(lengths)) > 1, retirement)
    hosts, host_ids, world_host_indices = [], {}, []
    for candidate, _ in worlds:
        if id(candidate.env) not in host_ids:
            host_ids[id(candidate.env)] = len(hosts)
            hosts.append(candidate.env)
        world_host_indices.append(host_ids[id(candidate.env)])
    if model_bank is None:
        packed = build_mixed_model(hosts, world_host_indices, device=device)
    else:
        bank_indices = {id(host): index for index, host in enumerate(model_bank.hosts)}
        packed = model_bank.bind([bank_indices[id(candidate.env)] for candidate, _ in worlds])
    model, warp_model = packed.carrier_model, packed.warp_model
    preserve_model_options(hosts[0]._model, warp_model)
    _check_noslip_support(hosts[0]._model, warp_model)
    packing_s = time.perf_counter() - started
    timings["mixed_model_upload_s"] = packing_s if model_bank is None else 0.0
    if model_bank is not None:
        timings["mixed_model_bind_s"] = packing_s
    upload_started = time.perf_counter()
    nworld = len(worlds)
    if model.nmocap != 1:
        raise MuJoCoWarpEvaluationError("V1.3 requires the original free-hand wrist mocap")
    from dexonomy.sim.eval_workspace import EvalWorkspace, eval_allocation
    allocation = eval_allocation(model, {k: cfg.get('eval_' + k, 0)
                                for k in ('njmax', 'nconmax', 'nccdmax')}, allocation_scale)
    if model_bank is not None:
        if not hasattr(model_bank, '_eval_workspace'):
            model_bank._eval_workspace = EvalWorkspace('floating-fc/v1')
        storage = model_bank._eval_workspace.acquire(
            (id(model), graph_key, tuple(sorted(allocation.items())), device),
            lambda: {'data': mjw.make_data(model, nworld=nworld, **allocation), 'graph': None},
            lambda slot: mjw.reset_data(warp_model, slot['data']))
        data = storage['data']
    else:
        storage = {'graph': None}
        data = mjw.make_data(model, nworld=nworld, **allocation)
    packed.initialize_data(data)
    rounds = np.asarray([len(candidate.control_qpos) for candidate, _ in worlds], dtype=np.int32)
    max_rounds = int(rounds.max())
    controls = np.zeros((nworld, max_rounds, model.nu), dtype=np.float32)
    positions = np.zeros((nworld, max_rounds, 3), dtype=np.float32)
    quaternions = np.zeros((nworld, max_rounds, 4), dtype=np.float32)
    initial_qpos, initial_ctrl, initial_pos, initial_quat = [], [], [], []
    body_ids, forces = [], []
    for world, (candidate, direction) in enumerate(worlds):
        env = candidate.env
        ctrl, pos, quat = _controls_for_worlds(env, candidate.control_qpos)
        controls[world, :rounds[world]] = ctrl
        positions[world, :rounds[world]] = pos[:, 0]
        quaternions[world, :rounds[world]] = quat[:, 0]
        initial_qpos.append(np.concatenate([candidate.initial_hand, env._obj_init_qpos]))
        ctrl, pos, quat = _controls_for_worlds(env, candidate.initial_hand[None])
        initial_ctrl.append(ctrl[0])
        initial_pos.append(pos[0])
        initial_quat.append(quat[0])
        body = env._obj_active_bodyid
        body_ids.append(body)
        forces.append(FORCE_DIRECTIONS[direction] * float(env._model.body_subtreemass[body] * 9.8))
    data.qpos.assign(np.asarray(initial_qpos, dtype=np.float32))
    data.ctrl.assign(np.asarray(initial_ctrl, dtype=np.float32))
    data.mocap_pos.assign(np.asarray(initial_pos, dtype=np.float32))
    data.mocap_quat.assign(np.asarray(initial_quat, dtype=np.float32))
    data.eq_active.assign(np.stack([candidate.env._model.eq_active0 for candidate, _ in worlds]))
    # make_data initializes integration state to zero, as native resetKeyframe.
    data.overflow.zero_()
    mjw.forward(warp_model, data)
    check_warp_allocation(data, mjw.OverflowType, context="V1.3 stability initial forward")

    clock = wp.zeros(1, dtype=int)
    preamble_rounds = wp.array(rounds, dtype=int)
    control_array = wp.array(controls, dtype=float)
    position_array = wp.array(positions, dtype=wp.vec3)
    quaternion_array = wp.array(quaternions, dtype=wp.quat)
    object_body = wp.array(np.asarray(body_ids, dtype=np.int32), dtype=int)
    force_array = wp.array(np.asarray(forces, dtype=np.float32), dtype=wp.vec3)
    active = wp.ones(nworld, dtype=int)
    newly_complete, completed = wp.zeros(nworld, dtype=int), wp.zeros(nworld, dtype=int)
    checkpoint_pos = wp.zeros((nworld, 10), dtype=wp.vec3)
    checkpoint_quat = wp.zeros((nworld, 10), dtype=wp.quat)
    checkpoint_seen = wp.zeros((nworld, 10), dtype=int)
    error_history = wp.zeros(nworld, dtype=int)
    retired = wp.zeros(nworld, dtype=int)
    solver_history = wp.zeros(nworld, dtype=int)
    placeholder_errors = wp.zeros(nworld, dtype=int)
    # Derived/contact/constraint workspaces are rebuilt by forward on every step.
    # Snapshots restore *all* integration/controller state before each placeholder
    # step; none of these placeholder results may enter checkpoints or feedback.
    safe = {name: wp.clone(getattr(data, name))
            for name in ResidentRefinementBatch.STATE} if retirement else {}
    terminal_qpos = np.full((nworld, model.nq), np.nan, dtype=np.float32)
    terminal_qvel = np.full((nworld, model.nv), np.nan, dtype=np.float32)
    terminal_steps = np.zeros(nworld, dtype=np.int32)
    retired_host = np.zeros(nworld, dtype=np.int32)
    unequal_preambles = bool(np.any(rounds != rounds[0]))
    saved = {
        name: wp.empty_like(getattr(data, name)) for name in ResidentRefinementBatch.STATE
    } if unequal_preambles else {}
    wp.synchronize_device(device)
    timings["state_upload_and_initial_forward_s"] = time.perf_counter() - upload_started

    def round_step():
        wp.launch(k.prepare_round, nworld, inputs=[clock, preamble_rounds, control_array,
            position_array, quaternion_array, active, data.ctrl, data.mocap_pos, data.mocap_quat])
        if retirement:
            wp.launch(k.apply_retirement, nworld, inputs=[retired, active])
        wp.launch(k.start_external_force, nworld, inputs=[clock, preamble_rounds,
            object_body, force_array, data.subtree_com, data.xipos, data.xfrc_applied])
        mjw.xfrc_accumulate(warp_model, data, data.qfrc_applied)
        data.xfrc_applied.zero_()
        def before_step():
            for name, snapshot in safe.items():
                copy_world_array(retired, snapshot, getattr(data, name))
        def after_step():
            wp.launch(k.accumulate_solver_flags, nworld, inputs=[active, data.overflow,
                int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS), solver_history])
            wp.launch(k.remember_errors, nworld, inputs=[retired, data.overflow,
                int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS),
                data.nacon, data.nefc, data.naconmax, data.njmax, data.qpos,
                data.qvel, placeholder_errors])
        checked_steps(warp_model, data, 5, active, error_history,
                      before_step=before_step if retirement else None,
                      after_step=after_step if retirement else None)
        wp.launch(k.record_checkpoint, nworld, inputs=[clock, preamble_rounds,
            object_body, data.xpos, data.xquat, checkpoint_pos, checkpoint_quat,
            checkpoint_seen, active, newly_complete, completed])
        for name, snapshot in saved.items():
            value = getattr(data, name)
            copy_world_array(newly_complete, value, snapshot)
            copy_world_array(completed, snapshot, value)
        wp.launch(k.advance_clock, 1, inputs=[clock])

    capture_started = time.perf_counter()
    workspace = storage["graph"] if reuse_graph else None
    arrays = [clock, preamble_rounds, control_array, position_array, quaternion_array,
              object_body, force_array, active, newly_complete, completed,
              checkpoint_pos, checkpoint_quat, checkpoint_seen, error_history,
              retired, solver_history, placeholder_errors]
    graph_hit = workspace is not None and workspace["data"] is data
    if graph_hit:
        # Graphs retain allocation addresses. Copy inputs into those exact buffers.
        for target, source in zip(workspace["arrays"], arrays):
            assert target.shape == source.shape and target.dtype == source.dtype
            wp.copy(target, source)
        for name, snapshot in safe.items():
            wp.copy(workspace["safe"][name], snapshot)
        import dataclasses
        old_model = workspace["model"]
        for field in dataclasses.fields(warp_model):
            source, target = getattr(warp_model, field.name), getattr(old_model, field.name)
            if isinstance(source, wp.array) and source.ptr != target.ptr:
                assert source.shape == target.shape
                wp.copy(target, source)
        if getattr(model_bank, "_native_expected", None) is not None:
            from dexonomy.sim.world_rows import verify_rows
            timings["graph_world_rows_checked"] = verify_rows(model_bank._native_expected,
                old_model, packed.world_host_indices)
        (clock, preamble_rounds, control_array, position_array, quaternion_array,
         object_body, force_array, active, newly_complete, completed,
         checkpoint_pos, checkpoint_quat, checkpoint_seen, error_history,
         retired, solver_history, placeholder_errors) = workspace["arrays"]
        capture = workspace["capture"]
        timings["graph_capture_s"] = 0.0
        timings["graph_rebind_s"] = time.perf_counter() - capture_started
    else:
        capture = capture_graph(round_step)
        timings["graph_capture_s"] = time.perf_counter() - capture_started
        if reuse_graph:
            storage["graph"] = {"arrays": arrays, "capture": capture,
                "data": data, "model": warp_model, "saved": saved, "safe": safe}
    timings["graph_cache_hit"] = int(graph_hit)
    physics_started = time.perf_counter()
    event_start, event_end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    wp.record_event(event_start)
    executed_rounds = 0
    for round_index in range(max_rounds + 100):
        launch_graph(capture)
        executed_rounds += 1
        if retirement:
            due = ((round_index + 1 - rounds) > 0) & ((round_index + 1 - rounds) <= 100) & (((round_index + 1 - rounds) % 10) == 0) & (retired_host == 0)
            if due.any():
                # Use the exact existing host metric (including its float32
                # quaternion convention), not a new approximate GPU threshold.
                if np.any(error_history.numpy()) or np.any(placeholder_errors.numpy()):
                    break
                from dexonomy.sim.world_retirement import inspect_checkpoint
                cp, cq = checkpoint_pos.numpy(), checkpoint_quat.numpy()
                newly_retired = []
                for world in np.flatnonzero(due):
                    hold = (round_index + 1 - rounds[world]) // 10 - 1
                    pose = np.concatenate([cp[world, hold], cq[world, hold]])
                    result = inspect_checkpoint(worlds[world][0].target_pose, pose,
                                                float(cfg.op.trans_thre), float(cfg.op.rot_thre))
                    if not result['passes'] or hold == 9:
                        newly_retired.append(world)
                if newly_retired:
                    qpos, qvel = data.qpos.numpy(), data.qvel.numpy()
                    for world in newly_retired:
                        terminal_qpos[world], terminal_qvel[world] = qpos[world], qvel[world]
                        terminal_steps[world] = (round_index + 1) * 5
                        retired_host[world] = 1
                    retired.assign(retired_host)
                if retired_host.all():
                    break
    wp.record_event(event_end)
    wp.synchronize_device(device)
    timings["physics_wall_s"] = time.perf_counter() - physics_started
    timings["physics_gpu_s"] = float(wp.get_event_elapsed_time(event_start, event_end)) / 1000.0
    download_started = time.perf_counter()
    if retirement and np.any(placeholder_errors.numpy()):
        raise MuJoCoWarpEvaluationError("retirement placeholder safety failure: " + str(placeholder_errors.numpy().tolist()))
    diagnostics = check_execution(data, error_history, "V1.3 mixed stability", typed_nonfinite=True)
    seen = checkpoint_seen.numpy()
    if not retirement and not np.all(seen == 1):
        raise MuJoCoWarpEvaluationError("V1.3 stability missed a per-world checkpoint")
    poses = np.concatenate([checkpoint_pos.numpy(), checkpoint_quat.numpy()], axis=2)
    if not np.isfinite(poses[seen.astype(bool)]).all():
        raise MuJoCoWarpEvaluationError("V1.3 stability non-finite checkpoint")
    if retirement:
        if not retired_host.all():
            raise MuJoCoWarpEvaluationError("missing retirement terminal state")
        poses[seen == 0] = np.nan  # Explicit NOT_RUN, never fabricated poses.
    timings["download_and_checks_s"] = time.perf_counter() - download_started
    return {
        "poses": poses,
        "solver_flags": solver_history.numpy() if retirement else diagnostics["solver_limit_flags"],
        **({"checkpoint_seen": seen, "terminal_qpos": terminal_qpos,
            "terminal_qvel": terminal_qvel, "terminal_steps": terminal_steps,
            "retirement_schema": "force_closure_6d_retirement_v1",
            "retired_worlds": int(retired_host.sum()),
            "placeholder_safety_flags": placeholder_errors.numpy()} if retirement else {}),
        "world_count": nworld,
        "active_worlds": nworld,
        "scene_count": len(hosts),
        "unique_scenes": len(hosts),
        "scene_paths": sorted({str(candidate.data["scene_path"]) for candidate, _ in worlds}),
        "candidate_count": len({id(candidate) for candidate, _ in worlds}),
        "duration_s": time.perf_counter() - started,
        "phase_timings": timings,
        "simulation_mj_steps": int(terminal_steps.sum()) if retirement else int(np.sum((rounds * 5 + 500))),
        "preamble_mj_steps": int(np.sum(rounds * 5)),
        "launched_world_mj_steps": int(nworld * executed_rounds * 5),
        "backend_metadata": {
            "backend": "mujoco_warp_mixed", "version": "V1.3.1" if model_bank is not None else "V1.3",
            "mujoco_version": mujoco.__version__, "mujoco_warp_version": mjw.__version__,
            "warp_version": wp.__version__, "device": device,
            "physics_profile": "native", "noslip_iterations": int(model.opt.noslip_iterations),
            "noslip_tolerance": float(model.opt.noslip_tolerance),
            "tolerance": float(model.opt.tolerance),
            "ccd_iterations": int(model.opt.ccd_iterations),
            "ccd_tolerance": float(model.opt.ccd_tolerance),
            "enableflags": int(model.opt.enableflags), "disableflags": int(model.opt.disableflags),
            "force_application": "preamble_derived_subtree_com_xfrc_accumulate_once",
            "allocation_diagnostics": diagnostics,
            "allocation": {**allocation, "nworld": nworld,
                "naconmax": int(data.naconmax), "naccdmax": int(data.naccdmax),
                "njmax_nnz": int(data.njmax_nnz)},
            "mixed_model": packed.metadata,
        },
    }


class MixedForceClosureBatcher:
    """Keep an incomplete 640-world batch across scene/epoch submissions."""

    def __init__(self, cfg, world_batch_size=640, *, scene_paths=None):
        if int(world_batch_size) < 1:
            raise ValueError("world_batch_size must be positive")
        if getattr(cfg, "physics_profile", "native") != "native":
            raise ValueError("V1.3 requires physics_profile='native'")
        if int(getattr(cfg.op, "noslip_iterations", 2)) not in (0, 2):
            raise ValueError("Supported explicit noslip_iterations values: 0, 2")
        self.cfg, self.world_batch_size = cfg, int(world_batch_size)
        self.pending = deque()
        self.environments = {}
        self.batch_index = 0
        self.model_builds = 0
        self.closed = False
        self.model_bank = None
        self.preparation_metrics = {}
        if scene_paths is not None:
            self._preload(scene_paths)

    def _make_scene_environment(self, scene_path):
        scene = _prepare_scene(scene_path, float(self.cfg.obj_mass))
        env = _make_environment(scene, str(self.cfg.hand.xml_path),
            tuple(float(value) for value in self.cfg.op.miu_coef),
            bool(self.cfg.op.hard), noslip_iterations=int(getattr(self.cfg.op, "noslip_iterations", 2)), physics_profile="native")
        env._model.opt.tolerance = 1e-8
        env._model.opt.noslip_tolerance = 1e-6
        if env._model.opt.noslip_iterations != int(getattr(self.cfg.op, "noslip_iterations", 2)):
            raise MuJoCoWarpEvaluationError("Model did not preserve explicitly requested noslip configuration")
        self.environments[scene_path] = (scene, env)
        self.model_builds += 1
        return scene, env

    def _preload(self, scene_paths):
        from dexonomy.sim.mujoco_warp_mixed import MixedModelBank

        started = time.perf_counter()
        scene_paths = tuple(dict.fromkeys(os.path.abspath(path) for path in scene_paths))
        if not scene_paths:
            raise ValueError("preloaded stability requires a nonempty scene inventory")
        for path in scene_paths:
            self._make_scene_environment(path)
        host_model_s = time.perf_counter() - started
        upload_started = time.perf_counter()
        self.model_bank = MixedModelBank([self.environments[path][1] for path in scene_paths],
            device=str(getattr(self.cfg, "warp_device", "cuda:0")))
        self.preparation_metrics = {
            **self.model_bank.metadata,
            "duration_s": time.perf_counter() - started,
            "host_model_s": host_model_s,
            "mixed_model_pack_upload_s": time.perf_counter() - upload_started,
            "host_model_builds": len(scene_paths),
            "carrier_model_builds": 1,
            "static_upload_count": 1,
            "scene_count": len(scene_paths),
        }

    @property
    def cache_stats(self):
        if self.model_bank is None:
            return {}
        workspace=getattr(self.model_bank,'_eval_workspace',None)
        return {**self.preparation_metrics, "bind_count": self.model_bank.bind_count,
                **(workspace.stats if workspace is not None else {})}

    def release_execution(self):
        """Drop Data/graphs before Refine while retaining immutable scene meshes."""
        if self.model_bank is not None and hasattr(self.model_bank, '_eval_workspace'):
            self.model_bank._eval_workspace.release()

    @property
    def pending_world_count(self):
        return len(self.pending)

    def _prepare(self, candidate, source_name):
        if candidate.get("pipeline_stage") != "stability_candidate":
            raise MuJoCoWarpEvaluationError("expected stability_candidate data")
        scene_path = str(candidate["scene_path"])
        if self.model_bank is not None:
            scene_path = os.path.abspath(scene_path)
            if scene_path not in self.environments:
                raise MuJoCoWarpEvaluationError(
                    f"scene was not in the preloaded stability inventory: {scene_path}")
        if scene_path not in self.environments:
            self._make_scene_environment(scene_path)
        scene, env = self.environments[scene_path]
        normalized = dict(candidate)
        if self.cfg.op.skip_pregrasp:
            normalized["pregrasp_qpos"] = candidate["grasp_qpos"].copy()
        trajectory = _candidate_trajectory(env, scene, normalized, bool(self.cfg.op.skip_pregrasp))
        return _Candidate(normalized, str(source_name), env, np.asarray(trajectory[2]),
            np.asarray(trajectory[0][0]), _round_controls(trajectory))

    def _run(self, count):
        worlds = [self.pending[index] for index in range(count)]
        from dexonomy.sim.mujoco_warp_env import MuJoCoWarpAllocationError
        audit_event("eval_batch_start",worlds=len(worlds),batch_index=self.batch_index+1)
        all_started = time.perf_counter()
        from .simulation_isolation import execute_eval_isolated, discard_batch_state
        original_worlds=list(worlds)
        self.simulation_error_ids=getattr(self,'simulation_error_ids',set())
        cleanup_s=0.0
        def simulate(active_worlds):
            kwargs={'allocation_scale':1}
            if self.model_bank is not None:kwargs['model_bank']=self.model_bank
            return _simulate_world_batch(active_worlds,self.cfg,**kwargs)
        def reset():
            self.release_execution()
            discard_batch_state(self.model_bank,str(getattr(self.cfg,'warp_device','cuda:0')))
        try:
            worlds,report,isolated,attempts=execute_eval_isolated(worlds,simulate,reset,self.cfg,self.simulation_error_ids)
        finally:
            cleanup_started=time.perf_counter()
            _release_batch_memory(str(getattr(self.cfg,'warp_device','cuda:0')))
            cleanup_s=time.perf_counter()-cleanup_started
        if report is None:report={'poses':np.empty((0,10,7)),'solver_flags':np.empty(0,dtype=int)}
        scale=1
        report['duration_all_attempts_s']=time.perf_counter()-all_started
        report['allocation_attempts']=attempts
        report.setdefault('phase_timings',{})['graph_pool_cleanup_s']=cleanup_s
        for _ in original_worlds:self.pending.popleft()
        isolated_ids={r['source_name'] for r in isolated}
        if isolated_ids:self.pending=deque((c,d) for c,d in self.pending if c.source_name not in isolated_ids)
        self.batch_index += 1
        report["batch_index"] = self.batch_index
        report["world_batch_size"] = self.world_batch_size
        report["allocation_scale"] = scale
        if worlds:
            audit_checkpoints(worlds,report["poses"],{k:v for k,v in report.items() if k!="poses"})
        else:
            audit_event('eval_batch_all_isolated',candidate_ids=sorted(isolated_ids),batch_index=self.batch_index)
        terminal_qpos_report = report.pop("terminal_qpos", None)
        terminal_qvel_report = report.pop("terminal_qvel", None)
        poses, flags = report.pop("poses"), report.pop("solver_flags")
        metrics_started = time.perf_counter()
        completed = list(isolated)
        for world, (candidate, direction) in enumerate(worlds):
            add_direction(candidate.checkpoint_poses, direction, poses[world])
            candidate.checkpoint_seen[direction] = np.asarray(report.get("checkpoint_seen", np.ones((len(worlds), 10), dtype=int))[world])
            if terminal_qpos_report is not None:
                count = int(candidate.checkpoint_seen[direction].sum())
                candidate.terminal_state[direction] = {
                    "checkpoint_pose": poses[world, count - 1].tolist(),
                    "integration_step": int(report["terminal_steps"][world]),
                    "qpos": terminal_qpos_report[world].tolist(),
                    "qvel": terminal_qvel_report[world].tolist(),
                    "pose_timing": "last_preintegration_derived_pose",
                    "integration_state_timing": "post_step_before_retirement_reset"}
            candidate.solver_flags[direction] = int(flags[world])
            candidate.batch_ids.add(self.batch_index)
            if len(candidate.checkpoint_poses) == len(FORCE_DIRECTIONS):
                ordered = np.stack(ordered_directions(candidate.checkpoint_poses))
                if self.cfg.get("eval_failed_world_retirement", False):
                    from dexonomy.sim.world_retirement import candidate_metrics
                    metrics = candidate_metrics(candidate.target_pose, ordered,
                        [candidate.checkpoint_seen[i] for i in range(6)], FORCE_DIRECTIONS,
                        float(self.cfg.op.trans_thre), float(self.cfg.op.rot_thre))
                else:
                    metrics = _metrics_from_checkpoints(candidate.target_pose,
                        [ordered[:, hold] for hold in range(10)],
                        float(self.cfg.op.trans_thre), float(self.cfg.op.rot_thre))[0]
                if self.cfg.get("eval_failed_world_retirement", False):
                    for direction, result in enumerate(metrics["direction_results"]):
                        result["terminal_checkpoint_pose"] = candidate.terminal_state[direction]["checkpoint_pose"]
                        if not result["passes"]:
                            result["terminal_state"] = candidate.terminal_state[direction]
                metrics.update({
                    "backend": "mujoco_warp_mixed", "physics_profile": "native",
                    "batch_indices": sorted(candidate.batch_ids),
                    "solver_limit_flags": [candidate.solver_flags[i] for i in range(6)],
                    "solver_limit_worlds": sum(bool(value) for value in candidate.solver_flags.values()),
                    "preamble_mj_steps": len(candidate.control_qpos) * 5 * 6,
                    "simulation_mj_steps": len(candidate.control_qpos) * 5 * 6 + int(sum(x.sum() for x in candidate.checkpoint_seen.values())) * 50,
                })
                success = bool(metrics["passes"])
                output = None
                if success:
                    output = dict(candidate.data)
                    output.update(pipeline_stage="stable_grasp", stability_metrics=metrics,
                        source_grasp_path=candidate.source_name)
                completed.append({"source_name": candidate.source_name, "success": success,
                    "output_data": output, "metrics": metrics,
                    "output_kind": "stable_grasp" if success else None})
        report["completed_candidate_count"] = len(completed)
        report["stable_count"] = sum(item["success"] for item in completed)
        report["phase_timings"]["metrics_and_result_assembly_s"] = time.perf_counter() - metrics_started
        return completed, report

    def push(self, candidates, *, source_names):
        if self.closed:
            raise RuntimeError("cannot push after flushing a V1.3 stability batcher")
        if len(candidates) != len(source_names):
            raise ValueError("candidate/source name lengths differ")
        started, builds_before = time.perf_counter(), self.model_builds
        cached_errors=[]
        for candidate, name in zip(candidates, source_names):
            from .simulation_isolation import require_finite_input
            if self.cfg.get('isolate_eval_nonfinite',False):require_finite_input(candidate)
            prepared = self._prepare(candidate, name)
            if self.cfg.get('isolate_eval_nonfinite',False):
                require_finite_input([prepared.initial_hand,prepared.control_qpos,prepared.target_pose,prepared.env._obj_init_qpos])
            if self.cfg.get('isolate_eval_epa',False) or self.cfg.get('isolate_eval_nonfinite',False):
                prepared.error_scene_metadata=dict(self.cfg.get('simulation_error_scene_metadata',{}).get(str(prepared.data['scene_path']),{}))
                from .simulation_isolation import known_error
                error=known_error(self.cfg.simulation_error_root,prepared)
                if error:
                    if error['metrics'].get('stage','eval')!='eval':raise RuntimeError('Eval ledger stage mismatch')
                    reason=error['metrics']['error']
                    if reason not in ('nonfinite','EPA_HORIZON') or not self.cfg.get('isolate_eval_nonfinite' if reason=='nonfinite' else 'isolate_eval_epa',False):raise RuntimeError('Unapproved cached simulation error')
                    prior=getattr(self,'simulation_error_ids',set());prior.add(prepared.source_name);self.simulation_error_ids=prior
                    if len(prior)>int(self.cfg.get('simulation_error_limit',1)):raise RuntimeError('Simulation error ledger limit exceeded')
                    cached_errors.append(error);continue
            self.pending.extend(direction_tasks([prepared]))
        preparation_s = time.perf_counter() - started
        results, groups = list(cached_errors), []
        while len(self.pending) >= self.world_batch_size:
            completed, report = self._run(self.world_batch_size)
            results.extend(completed)
            groups.append(report)
        return {"results": results, "groups": groups,
            "model_builds": self.model_builds - builds_before,
            "preparation_s": preparation_s, "pending_worlds": len(self.pending)}

    def flush(self):
        results, groups = [], []
        if self.pending:
            completed, report = self._run(len(self.pending))
            results.extend(completed)
            groups.append(report)
        self.closed = True
        return {"results": results, "groups": groups, "model_builds": 0,
            "preparation_s": 0.0, "pending_worlds": len(self.pending)}


def evaluate_force_closure_data_warp_mixed(candidates, cfg, *, source_names,
                                         world_batch_size=640):
    """Finite-input wrapper; streaming callers should retain the batcher."""
    batcher = MixedForceClosureBatcher(cfg, world_batch_size=world_batch_size)
    first = batcher.push(candidates, source_names=source_names)
    last = batcher.flush()
    return {"results": first["results"] + last["results"],
        "groups": first["groups"] + last["groups"],
        "model_builds": first["model_builds"], "preparation_s": first["preparation_s"],
        "pending_worlds": 0}
