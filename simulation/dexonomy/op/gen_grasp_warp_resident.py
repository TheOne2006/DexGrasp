"""Same-scene GPU-resident refinement with the unchanged boundary CPU QP."""
from __future__ import annotations
from dexonomy.acceptance_audit import event as audit_event

import os
import time
import gc
import logging

import numpy as np

from dexonomy.op.gen_grasp import GraspFilter, _normalize_grasp_data
from dexonomy.sim import HandCfg, MuJoCo_OptCfg, MuJoCo_OptEnv
from dexonomy.sim.mujoco_env import physics_stage_arena_memory_bytes
from dexonomy.util.file_util import load_scene_cfg
from dexonomy.util.np_util import np_array32


def _make_refinement_host(cfg, scene_path, *, mixed):
    return MuJoCo_OptEnv(
        hand_cfg=HandCfg(xml_path=cfg.hand.xml_path, freejoint=True),
        scene_cfg=load_scene_cfg(scene_path),
        sim_cfg=MuJoCo_OptCfg(obj_margin=cfg.op.pregrasp.cdist,
                             arena_memory_bytes=physics_stage_arena_memory_bytes(),
                             physics_profile="native" if mixed else getattr(cfg, "physics_profile", "upstream_mujoco333")),
        debug_render=False, debug_view=False)


class MixedRefinementCache:
    """V1.3.1: prepare every scene once and retain its static GPU model bank.

    Only batch state, candidate targets and world-to-scene indices change later.
    The bank snapshots the native initial transforms before boundary CPU filters
    overwrite a host's qpos/xpos/xmat, so those writes cannot leak into a batch.
    """

    def __init__(self, cfg, scene_paths):
        if getattr(cfg, "physics_profile", "native") != "native":
            raise ValueError("V1.3.1 requires physics_profile='native'")
        _validate_pregrasp_filter(cfg.op.pregrasp)
        if cfg.debug_render or cfg.debug_view or cfg.legacy_api:
            raise ValueError("preloaded refinement requires the non-debug V1.2 data API")
        from dexonomy.sim.mujoco_warp_mixed import MixedModelBank

        started = time.perf_counter()
        self.scene_paths = tuple(dict.fromkeys(os.path.abspath(path) for path in scene_paths))
        if not self.scene_paths:
            raise ValueError("preloaded refinement requires a nonempty scene inventory")
        self.path_indices = {path: index for index, path in enumerate(self.scene_paths)}
        self.hosts = [_make_refinement_host(cfg, path, mixed=True) for path in self.scene_paths]
        host_model_s = time.perf_counter() - started
        upload_started = time.perf_counter()
        self.bank = MixedModelBank(self.hosts, device=str(cfg.get("warp_device", "cuda:0")))
        self.preparation_metrics = {
            **self.bank.metadata,
            "duration_s": time.perf_counter() - started,
            "host_model_s": host_model_s,
            "mixed_model_pack_upload_s": time.perf_counter() - upload_started,
            "host_model_builds": len(self.hosts),
            "carrier_model_builds": 1,
            "static_upload_count": 1,
            "scene_count": len(self.hosts),
        }

    def close_admm_cpu_pool(self):
        """Release opt-in CPU workers before releasing this model bank."""
        pool = getattr(self, 'admm_cpu_pool', None)
        if pool is not None:
            pool.close()
            del self.admm_cpu_pool

    def close_traditional_qp_pool(self):
        """Release the optional traditional CPU-QP pool at bank shutdown."""
        pool = getattr(self, 'traditional_qp_pool', None)
        if pool is not None:
            pool.close()
            del self.traditional_qp_pool

    @property
    def cache_stats(self):
        return {**self.preparation_metrics, "bind_count": self.bank.bind_count}

    def world_indices(self, candidates):
        paths = [os.path.abspath(candidate["scene_path"]) for candidate in candidates]
        missing = sorted(set(paths).difference(self.path_indices))
        if missing:
            raise ValueError(f"scene was not in the preloaded refinement inventory: {missing}")
        return [self.path_indices[path] for path in paths]


def _contacts(host, packed, world):
    """Match the native contact extraction, including exported types and sign."""
    ho = {key: [] for key in ("dist", "pos", "normal", "bn1", "bn2")}
    hh = {key: [] for key in ho}
    hand_id = host.n_hand_body
    object_id = hand_id + host._model.nmocap
    for i in np.flatnonzero((packed["worldid"] == world) & ((packed["type"] & 5) != 0)):
        g1, g2 = packed["geom"][i]
        b1 = int(host._model.geom_bodyid[g1])
        b2 = int(host._model.geom_bodyid[g2])
        bn1, bn2 = host._model.body(b1).name, host._model.body(b2).name
        normal = packed["frame"][i].reshape(3, 3)[0]
        dist, pos = packed["dist"][i], packed["pos"][i]
        if (0 < b1 <= hand_id and b2 > object_id) or (0 < b2 <= hand_id and b1 > object_id):
            if b2 > object_id:
                hand_name, object_name = bn1.removeprefix(host._cfg.hand_prefix), bn2
            else:
                normal = -normal
                hand_name, object_name = bn2.removeprefix(host._cfg.hand_prefix), bn1
            ho["dist"].append(dist)
            ho["pos"].append(pos - dist * normal)
            ho["normal"].append(normal)
            ho["bn1"].append(hand_name)
            ho["bn2"].append(object_name)
        elif 0 < b1 <= hand_id and 0 < b2 <= hand_id:
            for name, value in (("dist", dist), ("pos", pos), ("normal", normal), ("bn1", bn1), ("bn2", bn2)):
                hh[name].append(value)
    return tuple({key: value if key.startswith("bn") else np_array32(value)
                  for key, value in contacts.items()} for contacts in (ho, hh))


def _update_template(data, cfg, host, local_contacts, contacts):
    mode = cfg.tmpl_upd_mode
    if mode == "orig":
        world_contacts = host.transform_cpn_b2w(data["hand_cbody"], local_contacts)
    elif mode in ("real", "disabled"):
        world_contacts = np.concatenate([contacts["pos"], contacts["normal"]], axis=-1)
        data["hand_cbody"] = contacts["bn1"]
    elif mode == "hybrid":
        world_contacts = host.transform_cpn_b2w(data["hand_cbody"], local_contacts)
        for body, point in zip(data["hand_cbody"], world_contacts):
            for actual_body, pos, normal in zip(contacts["bn1"], contacts["pos"], contacts["normal"]):
                if actual_body == body:
                    dist = np.linalg.norm(point[:3] - pos)
                    angle = np.arccos(np.clip((point[3:] * normal).sum(), -1, 1))
                    if dist < 0.03 and angle < np.pi / 4:
                        point[:3], point[3:] = pos, normal
                        break
    else:
        raise ValueError(f"Unsupported template update mode: {mode}")
    data["hand_cpn_w"] = world_contacts


def refine_grasp_batch(grasp_data_list, cfg, *, source_names=None, _mixed=False, world_batch_size=640,
                       _model_cache=None):
    """Replay an overflowing batch from its inputs with larger buffers only.

    No truncated state is accepted and no partially advanced state is resumed.
    Solver iterations, tolerances, margins and candidate budgets never change.
    Failed attempts and cleanup remain part of the reported total service time.
    """
    if not grasp_data_list:
        return []
    if _model_cache is not None and not _mixed:
        raise ValueError("a preloaded model cache requires mixed-object refinement")
    from dexonomy.sim.mujoco_warp_env import MuJoCoWarpAllocationError, release_warp_scene_memory

    audit_event("refine_batch_start",count=len(grasp_data_list),source_names=source_names)
    started = time.perf_counter()
    retries = []
    retry_elapsed = 0.0
    for scale in (1,):
        retry_error = None
        attempt_started = time.perf_counter()
        try:
            kwargs = {"source_names": source_names, "allocation_scale": scale}
            if _mixed:
                kwargs.update(mixed=True, world_batch_size=world_batch_size)
            if _model_cache is not None:
                kwargs["model_cache"] = _model_cache
            result = _refine_grasp_batch_once(grasp_data_list, cfg, **kwargs)
        except MuJoCoWarpAllocationError as error:
            retry_error = dict(error.diagnostics)
            if scale == 1:
                raise
        if retry_error is None:
            metrics = result[0]["metrics"]
            metrics["timings"]["batch_total_s"] = time.perf_counter() - started
            metrics["timings"]["allocation_retry_s"] = retry_elapsed
            metrics["model_builds"] = (0 if _model_cache is not None else
                (int(metrics.get("mixed_batch", {}).get("unique_scenes", 0)) + 1 if _mixed else 1) * (1 + len(retries)))
            metrics["allocation_scale"] = scale
            metrics["allocation_retries"] = retries
            return result
        # Leave the except block first so its traceback no longer owns the
        # failed model, device arrays and captured graphs.
        retries.append(retry_error)
        logging.warning("resident refinement retries complete batch at allocation scale %s: %s", scale * 2, retry_error)
        gc.collect()
        release_warp_scene_memory(cfg.get("warp_device", "cuda:0"))
        retry_elapsed += time.perf_counter() - attempt_started
    raise AssertionError("unreachable allocation retry loop")


def _validate_pregrasp_filter(pregrasp_cfg):
    """Reject filter branches that the resident final check cannot implement."""
    if not pregrasp_cfg:
        raise ValueError("resident refinement requires the V1.2 pregrasp stage")
    filters = pregrasp_cfg.get("filter")
    if not filters or not filters.get("limit") or not filters.get("collision"):
        raise ValueError("resident pregrasp requires enabled limit and collision filters")
    unsupported = sorted(name for name in filters
                         if name not in ("limit", "collision") and filters[name])
    if unsupported:
        raise ValueError("resident pregrasp supports only limit and collision filters; "
                         f"unsupported enabled filters: {', '.join(unsupported)}")


def _refine_grasp_batch_once(grasp_data_list, cfg, *, source_names=None, allocation_scale=1,
                             mixed=False, world_batch_size=640, model_cache=None):
    """Return native refinement result dictionaries, in the original order.

    Mixed batches keep one native host model per scene for unchanged boundary
    filters/QP; both simulation control loops stay on GPU.
    Batch costs are attached to the first result, not repeated per candidate.
    """
    if not grasp_data_list:
        return []
    _validate_pregrasp_filter(cfg.op.pregrasp)
    from dexonomy.sim.mujoco_warp_resident_env import ResidentRefinementBatch

    started = time.perf_counter()
    if cfg.debug_render or cfg.debug_view:
        raise ValueError("resident refinement requires debug_render/debug_view=False")
    if cfg.legacy_api:
        raise ValueError("resident scene batches require the V1.2 data API")
    data = [_normalize_grasp_data(item, cfg)[0] for item in grasp_data_list]
    paths = list(dict.fromkeys(os.path.abspath(x["scene_path"]) for x in data))
    if len(paths) != 1 and not mixed:
        raise ValueError("resident refinement batch contains different scenes")
    names = source_names if source_names is not None else [f"<memory:{i}>" for i in range(len(data))]
    if len(names) != len(data):
        raise ValueError("source_names length differs from candidate count")
    if mixed and len(data) > int(world_batch_size):
        raise ValueError("mixed refinement exceeds requested world batch capacity")
    if model_cache is None:
        hosts = [_make_refinement_host(cfg, path, mixed=mixed) for path in paths]
        world_indices = [paths.index(os.path.abspath(item["scene_path"])) for item in data]
        world_hosts = [hosts[index] for index in world_indices]
        host_model_s = time.perf_counter() - started
    else:
        world_indices = model_cache.world_indices(data)
        world_hosts = [model_cache.hosts[index] for index in world_indices]
        hosts = [model_cache.hosts[index] for index in dict.fromkeys(world_indices)]
        # CPU filters read only the exported qpos/xpos/xmat below. Restore the
        # native model margin; GPU initial state comes from the immutable bank.
        for scene_host in hosts:
            scene_host.set_obj_margin(cfg.op.pregrasp.cdist)
            scene_host.__dict__.pop("get_contacts", None)
        host_model_s = 0.0
    host = hosts[0]
    packed_model = None
    packing_started = time.perf_counter()
    if mixed:
        if model_cache is None:
            from dexonomy.sim.mujoco_warp_mixed import build_mixed_model
            packed_model = build_mixed_model(hosts, world_indices, device=str(cfg.get("warp_device", "cuda:0")))
        else:
            packed_model = model_cache.bank.bind(world_indices)
    packing_s = time.perf_counter() - packing_started if mixed else 0.0
    setup = time.perf_counter()
    batch_kwargs = {"allocation_scale": allocation_scale}
    if packed_model is not None:
        batch_kwargs["mixed"] = packed_model
    algorithm = str(cfg.get("refine_algorithm", "traditional"))
    if algorithm == "admm":
        from dexonomy.sim.mujoco_warp_admm import ADMMBatch
        batch_type = ADMMBatch
    elif algorithm == "traditional":
        batch_type = ResidentRefinementBatch
    else:
        raise ValueError(f"Unknown refinement algorithm: {algorithm}")
    batch = batch_type(host, data, cfg, **batch_kwargs)
    if algorithm == "admm":
        batch.admm_model_cache = model_cache
    batch.timings["host_model_s"] = host_model_s
    if mixed:
        batch.timings["mixed_model_pack_upload_s"] = packing_s if model_cache is None else 0.0
        if model_cache is not None:
            batch.timings["mixed_model_bind_s"] = packing_s
    batch.timings["device_setup_s"] = time.perf_counter() - setup
    exported = batch.refine()
    for scene_host in hosts:
        scene_host.set_obj_margin(cfg.op.grasp.cdist)
    results, passing, contact_sets = [], [], []
    filter_started = time.perf_counter()
    gpu_qp = str(cfg.get("refine_qp_backend", "cpu")) == "gpu"
    cpu_workers = int(cfg.get("refine_cpu_qp_workers", 0))
    parallel_qp = cpu_workers > 0 and algorithm == "traditional" and bool(cfg.op.grasp.filter.get('qp'))
    if cpu_workers < 0 or (parallel_qp and (gpu_qp or model_cache is None)):
        raise ValueError("parallel traditional QP requires CPU backend and an owned model cache")
    qp_jobs, qp_indices = [], []
    try:
        for i, item in enumerate(data):
            host = world_hosts[i]
            host._data.qpos[:] = exported["qpos"][i]
            host._data.xpos[:] = exported["xpos"][i]
            host._data.xmat[:] = exported["xmat"][i].reshape(-1, 9)
            contacts = (_contacts(host, packed_model.original_geoms(exported["contacts"], i), 0)
                        if packed_model is not None else _contacts(host, exported["contacts"], i))
            host.get_contacts = lambda _margin=None, value=contacts: value
            item["grasp_qpos"] = np_array32(host.get_hand_qpos())[None]
            filt = GraspFilter(cfg.op.grasp.filter, host, names[i] + " grasp")
            qp_original = filt._qp_filter
            def timed_qp(*args, _original=qp_original):
                begin = time.perf_counter()
                result = _original(*args)
                batch.timings["cpu_qp_s"] += time.perf_counter() - begin
                return result
            filt._qp_filter = timed_qp
            try:
                if gpu_qp or parallel_qp:
                    # Run the unchanged earlier filters first, deferring only QP.
                    filt._qp_filter = lambda *args: True
                passed, ho = filt.forward(item)
                if (gpu_qp or parallel_qp) and passed:
                    qp_indices.append(i)
                    qp_jobs.append((ho["pos"].copy(), ho["normal"].copy(),
                                    item["ext_wrench"].copy(), item["ext_center"].copy()))
            finally:
                # Do not retain a bound-method cycle holding the GPU batch.
                del filt.__dict__["_qp_filter"]
            item["ho_c"] = ho
            passing.append(passed)
            contact_sets.append(ho if passed else None)
            metrics = {"passes": False, "failure_stage": "grasp", "grasp_qpos": item["grasp_qpos"][0].tolist(),
                       "grasp_iterations": int(batch.grasp_iterations[i]), "pregrasp_iterations": 0}
            results.append({"success": False, "output_data": None, "metrics": metrics})
            if passed and not parallel_qp:
                count = len(item["hand_cbody"])
                local = np.concatenate((batch.local_host[i, :count], batch.local_normals_host[i, :count]), axis=-1)
                _update_template(item, cfg, host, local, ho)
    finally:
        for scene_host in hosts:
            scene_host.__dict__.pop("get_contacts", None)
    batch.timings["cpu_filters_and_template_s"] = time.perf_counter() - filter_started
    if parallel_qp and qp_jobs:
        from dexonomy.qp.traditional_pool import TraditionalQPPool, solve_job
        backend = str(cfg.get('refine_cpu_qp_executor', 'thread'))
        pool = getattr(model_cache, 'traditional_qp_pool', None)
        if pool is not None and (pool.workers != cpu_workers or pool.backend != backend):
            model_cache.close_traditional_qp_pool()
            pool = None
        if pool is None:
            pool = model_cache.traditional_qp_pool = TraditionalQPPool(cpu_workers, backend)
        jobs = [(*job, list(cfg.op.grasp.filter.qp.miu_coef)) for job in qp_jobs]
        begin = time.perf_counter()
        solved = pool.solve(jobs)
        batch.timings['cpu_qp_s'] = time.perf_counter() - begin
        if cfg.get('refine_verify_cpu_qp', False):
            begin = time.perf_counter()
            for job, (wrench, error) in zip(jobs, solved):
                reference, reference_error = solve_job(job)
                assert (wrench is None) == (reference is None)
                assert error == reference_error
                if wrench is not None:
                    assert np.array_equal(wrench, reference)
            batch.timings['cpu_qp_verify_s'] = time.perf_counter() - begin
            audit_event('traditional_qp_verified', count=len(jobs), workers=cpu_workers, backend=backend)
        begin = time.perf_counter()
        for i, (wrench, error) in zip(qp_indices, solved):
            passed = not (error > cfg.op.grasp.filter.qp.thre or wrench is None)
            passing[i] = passed
            if not passed:
                contact_sets[i] = None
                continue
            item, host, ho = data[i], world_hosts[i], contact_sets[i]
            ho['wrench'] = 10 * wrench
            # Restore this candidate before template transforms; host models are shared.
            host._data.qpos[:] = exported['qpos'][i]
            host._data.xpos[:] = exported['xpos'][i]
            host._data.xmat[:] = exported['xmat'][i].reshape(-1, 9)
            count = len(item['hand_cbody'])
            local = np.concatenate((batch.local_host[i, :count], batch.local_normals_host[i, :count]), axis=-1)
            _update_template(item, cfg, host, local, ho)
        batch.timings['cpu_qp_postprocess_s'] = time.perf_counter() - begin
    if gpu_qp and qp_jobs:
        from dexonomy.qp.refine_batched import solve_contacts
        begin = time.perf_counter()
        solved = solve_contacts(qp_jobs, cfg.op.grasp.filter.qp.miu_coef,
                                device=str(cfg.get("warp_device", "cuda:0")))
        batch.timings["gpu_qp_s"] = time.perf_counter() - begin
        qp_postprocess_started = time.perf_counter()
        for i, job, answer in zip(qp_indices, qp_jobs, solved):
            passed = answer["finite"] and answer["error"] <= cfg.op.grasp.filter.qp.thre
            passing[i] = passed
            if passed:
                data[i]["ho_c"]["wrench"] = 10 * answer["wrench"]
            else:
                contact_sets[i] = None
            results[i]["metrics"]["qp"] = {k:v for k,v in answer.items() if k not in ("wrench", "parameters")}
            audit_event("refine_gpu_qp", source=names[i], **results[i]["metrics"]["qp"])
        if os.environ.get("DEXONOMY_QP_CAPTURE"):
            from pathlib import Path
            dest=Path(os.environ["DEXONOMY_QP_CAPTURE"]);dest.mkdir(parents=True,exist_ok=True)
            np.save(dest/f"qp-{time.time_ns()}.npy", {"records":qp_jobs,"results":solved,
                "names":[names[i] for i in qp_indices],"miu_coef":list(cfg.op.grasp.filter.qp.miu_coef),
                "threshold":float(cfg.op.grasp.filter.qp.thre)})
        batch.timings["qp_postprocess_and_capture_s"] = time.perf_counter() - qp_postprocess_started
    if any(passing):
        pre = batch.pregrasp(passing, contact_sets)
        for i, passed in enumerate(passing):
            if not passed:
                continue
            host = world_hosts[i]
            metrics = results[i]["metrics"]
            metrics["pregrasp_iterations"] = int(pre["iterations"][i])
            host._data.qpos[:] = pre["qpos"][i]
            collision = cfg.op.pregrasp.filter.collision
            valid = (host.check_qpos_limit(cfg.op.pregrasp.filter.limit.thre)
                     and pre["hh"][i] >= collision.hh_thre and pre["ho"][i] >= collision.ho_thre)
            if not valid:
                metrics["failure_stage"] = "pregrasp"
                continue
            item = data[i]
            item["squeeze_qpos"] = np_array32(pre["squeeze"][i])[None]
            item["pregrasp_qpos"] = np_array32(pre["trace"][i, :pre["iterations"][i]][::-2])
            item["n_evo"] = np_array32(item["n_evo"] + 1)
            metrics.update(passes=True, failure_stage=None)
            results[i].update(success=True, output_data=item)
    if algorithm == "admm":
        for result, diagnostic in zip(results, batch.admm_diagnostics):
            result["metrics"]["admm"] = diagnostic
            if diagnostic["fallback"]:
                # The public ADMM wrapper reruns just these inputs with the
                # official 20-step fallback, retaining both traces.
                result.update(success=False, output_data=None)
                result["metrics"]["failure_stage"] = "admm_qp_fallback_pending"
    for result in results:
        metrics = result["metrics"]
        metrics["simulation_mj_steps"] = (
            int(metrics["grasp_iterations"]) * int(cfg.op.grasp.substep)
            + int(metrics["pregrasp_iterations"]) * int(cfg.op.pregrasp.substep)
        )
    batch.finish_timings()
    batch.timings["batch_total_s"] = time.perf_counter() - started
    results[0]["metrics"].update(timings=dict(batch.timings), model_builds=1,
                                 batch_size=len(data), profiling=batch.profile)
    if packed_model is not None:
        metadata = dict(packed_model.metadata)
        metadata.update(world_batch_capacity=int(world_batch_size), phase_timings=dict(batch.timings),
                        world_scene_indices=world_indices,
                        grasp_effective_world_steps=int(sum(batch.grasp_iterations)) * int(cfg.op.grasp.substep),
                        grasp_launched_world_steps=len(data) * int(cfg.op.grasp.step) * int(cfg.op.grasp.substep) * (int(cfg.op.grasp.step_hand) if algorithm == "admm" else 1),
                        pregrasp_effective_world_steps=sum(r["metrics"]["pregrasp_iterations"] for r in results) * int(cfg.op.pregrasp.substep),
                        pregrasp_launched_world_steps=(len(data) * int(cfg.op.pregrasp.step) * int(cfg.op.pregrasp.substep) if any(passing) else 0))
        results[0]["metrics"]["mixed_batch"] = metadata
    return results


def _refine_grasp_batch_mixed_once(grasp_data_list, cfg, *, source_names=None, world_batch_size=640,
                             model_cache=None):
    """V1.3: independent object scenes in a resident Warp world batch."""
    minimum=int(cfg.get("refine_min_execution_slots",1))
    names=source_names if source_names is not None else [f"<memory:{i}>" for i in range(len(grasp_data_list))]
    from dexonomy.night_contracts import padded_inputs
    physical,physical_names,indices=padded_inputs(grasp_data_list,names,minimum,world_batch_size)
    audit_event("refine_execution_padding",logical_count=len(names),execution_count=len(indices),logical_names=names,execution_names=physical_names,original_indices=indices)
    from .simulation_isolation import RefinementNonfiniteError
    try:
        results=refine_grasp_batch(physical,cfg,source_names=physical_names,_mixed=True,world_batch_size=world_batch_size,_model_cache=model_cache)
    except RefinementNonfiniteError as error:
        error.world_indices=indices
        raise
    assert len(results)==len(indices)
    # Never choose a successful duplicate in place of its original slot.
    selected=results[:len(names)]
    audit_event("refine_padding_discarded",logical_names=names,returned_count=len(selected),discarded_names=physical_names[len(names):],discarded_labels=[bool(r["success"]) for r in results[len(names):]])
    return selected


def refine_grasp_batch_mixed(grasp_data_list,cfg,*,source_names=None,world_batch_size=640,model_cache=None,simulation_error_ids=None):
    from .simulation_isolation import execute_refine_isolated,discard_batch_state
    names=list(source_names) if source_names is not None else [f'<memory:{i}>' for i in range(len(grasp_data_list))]
    def execute(data,ids):
        return _refine_grasp_batch_mixed_once(data,cfg,source_names=ids,world_batch_size=world_batch_size,model_cache=model_cache)
    def reset():
        gc.collect()
        discard_batch_state(model_cache.bank if model_cache is not None else None,str(cfg.get('warp_device','cuda:0')))
    return execute_refine_isolated(grasp_data_list,names,execute,reset,cfg,set() if simulation_error_ids is None else simulation_error_ids)
