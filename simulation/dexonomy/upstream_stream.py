"""V1.3.1 shared-memory pipeline terminating at six-direction stability."""

from __future__ import annotations
from dexonomy.acceptance_audit import event as audit_event, sample as audit_sample

import argparse
import json
import importlib.metadata
import logging
import multiprocessing
import os
import queue
import random
import subprocess
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager, nullcontext
from itertools import chain
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from hydra import compose, initialize_config_module

from dexonomy.upstream_batch import _case_map, _load_index, _scene_id_from_data, convert_batch, build_batch_report_from_summaries
from dexonomy.upstream_pipeline import PipelineError, _load_dict, _stable_score
from dexonomy.util.cpu_slots import PriorityCpuSlots, detect_available_cpu_slots
from dexonomy.util.file_util import load_scene_cfg
from dexonomy.util.scene_shm import (
    cleanup_scene_shm_namespace,
    consume_scene_bundle,
    publish_scene_bundle,
)


_POOL_CFG = None
_POOL_STAGE = None
_POOL_INPUT_ROOT = None
_POOL_SUCCESS_ROOT = None
_SHM_NAMESPACE = None
_SHM_ROOTS = None
_CPU_SLOTS = None
_CPU_CANCEL_EVENT = None
_TEMPLATE_FEEDBACK_QUEUE = None
_TEMPLATE_FEEDBACK_ACCEPTING_EVENT = None
_TEMPLATE_FEEDBACK_MAX_EVO = 2

_TEMPLATE_FEEDBACK_FORMAT = "dexonomy-template-feedback/v1"
_TEMPLATE_FEEDBACK_FIELDS = (
    "grasp_qpos",
    "hand_cpn_w",
    "hand_cbody",
    "required_cbody",
    "tmpl_name",
    "n_evo",
)


def _signal(path: Path, stage: str, processed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"stage": stage, "processed": processed, "completed_epoch_s": time.time()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )



def _compose_cfg(overrides: list[str]):
    with initialize_config_module(
        config_module="dexonomy.config", version_base=None
    ):
        cfg = compose(config_name="base", overrides=overrides)
        from omegaconf import OmegaConf
        audit_event("resolved_config",overrides=overrides,config=OmegaConf.to_container(cfg,resolve=True))
        return cfg



def _shm_pool_initializer(
    cfg,
    stage: str,
    namespace: str,
    logical_roots: dict[str, str],
    cpu_slots=None,
    cancel_event=None,
    template_feedback_queue=None,
    template_feedback_accepting_event=None,
    template_feedback_max_evo=2,
) -> None:
    """Initialize a scene worker that consumes shared-memory bundles."""

    global _POOL_CFG, _POOL_STAGE, _SHM_NAMESPACE, _SHM_ROOTS
    global _CPU_SLOTS, _CPU_CANCEL_EVENT
    global _TEMPLATE_FEEDBACK_QUEUE, _TEMPLATE_FEEDBACK_ACCEPTING_EVENT
    global _TEMPLATE_FEEDBACK_MAX_EVO
    _POOL_CFG = cfg
    _POOL_STAGE = stage
    _SHM_NAMESPACE = namespace
    _SHM_ROOTS = {key: Path(value) for key, value in logical_roots.items()}
    _CPU_SLOTS = cpu_slots
    _CPU_CANCEL_EVENT = cancel_event
    _TEMPLATE_FEEDBACK_QUEUE = template_feedback_queue
    _TEMPLATE_FEEDBACK_ACCEPTING_EVENT = template_feedback_accepting_event
    _TEMPLATE_FEEDBACK_MAX_EVO = int(template_feedback_max_evo)



def _publish_template_feedback(
    scene_id: str,
    relative: Path,
    refined_data: dict[str, Any],
) -> bool:
    """Feed a stable refined grasp back to init using DexSyn's template fields."""

    if _TEMPLATE_FEEDBACK_QUEUE is None:
        return False
    if (
        _TEMPLATE_FEEDBACK_ACCEPTING_EVENT is None
        or not _TEMPLATE_FEEDBACK_ACCEPTING_EVENT.is_set()
    ):
        return False
    missing = [key for key in _TEMPLATE_FEEDBACK_FIELDS if key not in refined_data]
    if missing:
        raise PipelineError(
            f"stable candidate {relative} lacks template fields: {missing}"
        )
    n_evo = float(np.asarray(refined_data["n_evo"]).reshape(-1)[0])
    if n_evo > _TEMPLATE_FEEDBACK_MAX_EVO:
        return False
    compact_data = {}
    for key in _TEMPLATE_FEEDBACK_FIELDS:
        value = refined_data[key]
        compact_data[key] = value.copy() if isinstance(value, np.ndarray) else value
    envelope = {
        "format_version": _TEMPLATE_FEEDBACK_FORMAT,
        "template_id": f"{scene_id}:{relative.as_posix()}",
        "data": compact_data,
    }
    while _TEMPLATE_FEEDBACK_ACCEPTING_EVENT.is_set():
        if _CPU_CANCEL_EVENT is not None and _CPU_CANCEL_EVENT.is_set():
            raise PipelineError("template feedback was cancelled")
        try:
            _TEMPLATE_FEEDBACK_QUEUE.put(envelope, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False



def _shm_candidate(entry: Any, scene_id: str) -> tuple[Path, dict[str, Any]]:
    """Validate one internal candidate envelope and its logical relative path."""

    if not isinstance(entry, dict):
        raise PipelineError(f"shared-memory candidate is not a dictionary: {entry!r}")
    relative_raw = entry.get("relative_path")
    data = entry.get("data")
    if not isinstance(relative_raw, str) or not relative_raw:
        raise PipelineError("shared-memory candidate has no relative_path")
    relative = Path(relative_raw)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".npy":
        raise PipelineError(f"unsafe shared-memory candidate path: {relative_raw!r}")
    if len(relative.parts) < 2 or relative.parts[-2] != scene_id:
        raise PipelineError(
            f"candidate {relative_raw!r} does not belong to scene {scene_id!r}"
        )
    if not isinstance(data, dict):
        raise PipelineError(f"candidate {relative_raw!r} has no data dictionary")
    return relative, data



def _shm_case(bundle: dict[str, Any], scene_id: str) -> dict[str, Any]:
    case = bundle.get("metadata", {}).get("case")
    if not isinstance(case, dict) or case.get("scene_id") != scene_id:
        raise PipelineError(f"shared-memory scene {scene_id!r} has invalid case metadata")
    for key in ("stability_scene",):
        if not isinstance(case.get(key), str):
            raise PipelineError(f"shared-memory scene {scene_id!r} lacks {key}")
    return case



def _validate_shm_stability_scene(case: dict[str, Any]) -> None:
    stability_path = Path(case["stability_scene"])
    scene = load_scene_cfg(str(stability_path))
    target_name = scene.get("task", {}).get("obj_name")
    entities = scene.get("scene", {})
    if (
        scene.get("pipeline_stage") != "grasp_stability"
        or scene.get("task", {}).get("type") != "force_closure"
        or set(entities) != {target_name}
        or entities[target_name].get("type") != "rigid_object"
        or entities[target_name].get("fix_root", False)
    ):
        raise PipelineError(f"invalid isolated stability scene: {stability_path}")



def _result_metric_fields(metrics: Any) -> dict[str, int]:
    if not isinstance(metrics, dict):
        return {}
    return {
        "simulation_mj_steps": int(metrics.get("simulation_mj_steps", 0)),
        "preamble_mj_steps": int(metrics.get("preamble_mj_steps", 0)),
    }



def _new_stage_summary(
    stage: str,
    task_unit: str = "candidate",
    transport: str = "queue",
) -> dict[str, Any]:
    return {
        "format_version": "dexonomy-online-stage-summary/v1",
        "transport": transport,
        "task_unit": task_unit,
        "stage": stage,
        "tasks": 0,
        "model_builds": 0,
        "simulation_mj_steps": 0,
        "preamble_mj_steps": 0,
        "processed": 0,
        "success": 0,
        "first_success_epoch_s": None,
        "last_success_epoch_s": None,
        "scene_bundles_in": 0,
        "scene_bundles_out": 0,
        "shared_memory_bytes_in": 0,
        "shared_memory_bytes_out": 0,
        "template_feedback_published": 0,
        "failure_stages": {},
        "epochs": {},
        "per_scene": {},
    }



def _record_stage_result(
    summary: dict[str, Any],
    scene_id: str,
    success: bool,
    result: dict[str, Any] | None = None,
) -> None:
    summary["processed"] += 1
    scene = summary["per_scene"].setdefault(
        scene_id, {"processed": 0, "success": 0}
    )
    scene["processed"] += 1
    if result is not None:
        summary["simulation_mj_steps"] += int(
            result.get("simulation_mj_steps", 0)
        )
        summary["preamble_mj_steps"] += int(result.get("preamble_mj_steps", 0))
        if isinstance(result.get("timings"), dict):
            for key, seconds in result["timings"].items():
                timings = summary.setdefault("timings", {})
                timings[key] = float(timings.get(key, 0.0)) + float(seconds)
    if not success:
        failure_stage = result.get("failure_stage") if result is not None else None
        if isinstance(failure_stage, str):
            summary["failure_stages"][failure_stage] = (
                int(summary["failure_stages"].get(failure_stage, 0)) + 1
            )
            scene_failure_stages = scene.setdefault("failure_stages", {})
            scene_failure_stages[failure_stage] = (
                int(scene_failure_stages.get(failure_stage, 0)) + 1
            )
        return
    now = time.time()
    summary["success"] += 1
    scene["success"] += 1
    if summary["first_success_epoch_s"] is None:
        summary["first_success_epoch_s"] = now
    summary["last_success_epoch_s"] = now


def _record_epoch_bundle(
    summary: dict[str, Any],
    epoch: int,
    results: list[dict[str, Any]],
    *,
    models_built: int = 0,
    templates_published: int = 0,
) -> None:
    epoch_summary = summary["epochs"].setdefault(
        str(epoch),
        {
            "tasks": 0,
            "model_builds": 0,
            "processed": 0,
            "success": 0,
            "template_feedback_published": 0,
        },
    )
    epoch_summary["tasks"] += 1
    epoch_summary["model_builds"] += int(models_built)
    epoch_summary["processed"] += len(results)
    epoch_summary["success"] += sum(
        1 for result in results if bool(result.get("success"))
    )
    epoch_summary["template_feedback_published"] += int(templates_published)



def _write_stage_summary(signal_dir: Path, summary: dict[str, Any]) -> Path:
    path = signal_dir / f"{summary['stage']}.summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path



def _common_overrides(args, op: str, exp_name: str) -> list[str]:
    return [
        f"op={op}",
        f"hand={args.hand}",
        f"op_name={op}",
        f"hand_name={args.hand}",
        f"exp_name={exp_name}",
        f"asset_root={args.asset_root}",
        f"save_root={args.save_root}",
        f"n_worker={args.workers}",
        "skip_done=False",
        "debug_render=False",
        "debug_view=False",
    ]



def _consume_shm_queue(input_queue, cancel_event=None):
    """Yield one single-owner scene descriptor at a time until the sentinel."""

    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise PipelineError("shared-memory pipeline was cancelled")
        try:
            item = input_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if item is None:
            return
        if not isinstance(item, dict) or item.get("format_version") != (
            "dexonomy-scene-shm/v1"
        ):
            raise PipelineError(f"invalid shared-memory queue item: {item!r}")
        yield item



def _shm_queue_put(output_queue, item, cancel_event) -> None:
    """Bounded, cancellable queue send used at shared-memory boundaries."""

    wait_started = time.perf_counter()
    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise PipelineError("shared-memory pipeline was cancelled")
        try:
            output_queue.put(item, timeout=0.5)
            audit_event("ipc_put",wall_s=time.perf_counter()-wait_started,size=item.get("size",0) if isinstance(item,dict) else 0)
            return
        except queue.Full:
            continue



def _process_shm_mixed_descriptors(
    descriptors, *, stage, cfg, workers, summary, namespace, logical_roots,
    output_queue=None, cancel_event=None, cpu_slots=None,
    template_feedback_queue=None, template_feedback_accepting_event=None,
    template_feedback_max_evo=2, pool_start_method="spawn", scene_paths=None,
) -> None:
    """Fill GPU world batches across scene envelopes, then restore the envelopes.

    Scene/epoch metadata, candidate ordering, filtering and template publication
    use the existing postprocessing path.  No epoch barrier is introduced.  Only
    end-of-stream flushes the final incomplete world batch.
    """
    _shm_pool_initializer(
        cfg, stage, namespace, logical_roots, cpu_slots, cancel_event,
        template_feedback_queue, template_feedback_accepting_event,
        template_feedback_max_evo,
    )
    capacity = int(cfg.warp_world_batch_size)
    if capacity < 1:
        raise PipelineError("world batch size must be positive")
    records = deque()
    pending = deque()
    routes = {}
    batches = summary.setdefault("world_batches", [])
    summary["world_batch_capacity"] = capacity
    summary["world_semantics"] = "candidate" if stage == "grasp" else "candidate_force_direction"
    summary["physics_profile"] = str(cfg.physics_profile)
    summary["batching_policy"] = "cross_scene_fifo_full_batches_flush_only_at_eof"
    stability_batcher = None
    refinement_cache = None
    if stage not in {"grasp", "stability"}:
        raise PipelineError("mixed world batching only supports grasp/stability")
    cache_mode = getattr(cfg, "warp_model_cache", "none")
    if cache_mode not in {"none", "preloaded"}:
        raise PipelineError(f"unknown model cache mode: {cache_mode}")
    summary["model_cache"] = {"mode": cache_mode}

    @contextmanager
    def cpu_lease():
        waiting = time.perf_counter()
        lease = (cpu_slots.lease(stage, cancel_event=cancel_event)
                 if cpu_slots is not None else nullcontext())
        with lease:
            summary["cpu_slot_wait_s"] = summary.get("cpu_slot_wait_s", 0.) + time.perf_counter() - waiting
            yield

    def record_service(duration, component):
        components = summary.setdefault("service_components_s", {})
        components[component] = float(components.get(component, 0.)) + duration
        timing = summary.setdefault("service_duration_s", {"sum": 0., "max": 0., "mean": 0.})
        timing["sum"] += duration
        timing["max"] = max(timing["max"], duration)
        timing["mean"] = timing["sum"] / max(1, summary["tasks"])

    preparation_duration = None

    def update_cache_stats():
        cache = refinement_cache if stage == "grasp" else stability_batcher
        if cache_mode == "preloaded":
            stats = dict(cache.cache_stats)
            summary["model_cache"] = {
                **stats, "mode": cache_mode,
                "internal_duration_s": stats.get("duration_s"),
                "duration_s": preparation_duration,
            }

    # Build the complete inventory before consuming any scene envelope. The
    # initializer may run concurrently; the original queue limits still apply.
    if cache_mode == "preloaded":
        if not scene_paths:
            raise PipelineError("preloaded model cache requires the complete scene inventory")
        with cpu_lease():
            preparation_started = time.perf_counter()
            if stage == "grasp":
                from dexonomy.op.gen_grasp_warp_resident import MixedRefinementCache
                refinement_cache = MixedRefinementCache(cfg, scene_paths)
            else:
                from dexonomy.op.eval_force_closure_warp_mixed import MixedForceClosureBatcher
                stability_batcher = MixedForceClosureBatcher(
                    cfg, world_batch_size=capacity, scene_paths=scene_paths,
                )
            preparation_duration = time.perf_counter() - preparation_started
        update_cache_stats()
        audit_event("model_bank",stage=stage,stats=summary["model_cache"])
        summary["model_builds"] += (summary["model_cache"]["host_model_builds"]
                                    + summary["model_cache"]["carrier_model_builds"])
        record_service(preparation_duration, "model_preparation")
        print(f"[shm:{stage}] preloaded {summary['model_cache']['scene_count']} models "
              f"in {preparation_duration:.3f}s; static assets resident on GPU", flush=True)
    elif stage == "stability":
        from dexonomy.op.eval_force_closure_warp_mixed import MixedForceClosureBatcher
        stability_batcher = MixedForceClosureBatcher(cfg, world_batch_size=capacity)

    def accept_ready():
        while records and all(item is not None for item in records[0]["results"]):
            record = records.popleft()
            with cpu_lease():
                grouped = _process_shm_scene_impl(
                    record["descriptor"], bundle=record["bundle"],
                    precomputed_results=record["results"],
                )
            summary["tasks"] += 1
            summary["scene_bundles_in"] += 1
            summary["shared_memory_bytes_in"] += int(grouped["input_size"])
            summary["template_feedback_published"] += int(grouped["templates_published"])
            summary["model_builds"] += int(grouped["models_built"])
            # Include shared batch cost exactly once, then add each scene's
            # postprocessing. Mean is amortized per scene/epoch envelope.
            record_service(float(grouped["service_duration_s"]), "scene_postprocessing")
            _record_epoch_bundle(
                summary, int(grouped["epoch"]), grouped["results"],
                models_built=int(grouped["models_built"]),
                templates_published=int(grouped["templates_published"]),
            )
            for result in grouped["results"]:
                _record_stage_result(summary, result["scene_id"], result["success"], result)
            descriptor = grouped["output_descriptor"]
            if descriptor is not None:
                summary["scene_bundles_out"] += 1
                summary["shared_memory_bytes_out"] += int(descriptor["size"])
                _shm_queue_put(output_queue, descriptor, cancel_event)

    def accept_stability(report):
        summary["model_builds"] += int(report.get("model_builds", 0))
        timings = summary.setdefault("timings", {})
        timings["candidate_and_model_preparation_s"] = (
            timings.get("candidate_and_model_preparation_s", 0.) + float(report.get("preparation_s", 0.)))
        for group in report.get("groups", []):
            batches.append({"batch_id": len(batches), "world_capacity": capacity, **group})
            for key, seconds in group.get("phase_timings", {}).items():
                timings = summary.setdefault("timings", {})
                timings[key] = float(timings.get(key, 0.)) + float(seconds)
            print(f"[shm:{stage}] mixed world batch {len(batches)} complete", flush=True)
        for result in report.get("results", []):
            source = result.get("source_name")
            if source not in routes:
                raise PipelineError(f"unknown or duplicate mixed stability result: {source}")
            record, offset = routes.pop(source)
            record["results"][offset] = result
        accept_ready()

    def run_refinement(count):
        from dexonomy.op.gen_grasp_warp_resident import refine_grasp_batch_mixed
        from dexonomy.sim.mujoco_warp_env import release_warp_scene_memory
        entries = [pending.popleft() for _ in range(count)]
        with cpu_lease():
            started = time.perf_counter()
            try:
                values = refine_grasp_batch_mixed(
                    [data for _record, _offset, _name, data in entries], cfg,
                    source_names=[name for _record, _offset, name, _data in entries],
                    world_batch_size=capacity,
                    **({"model_cache": refinement_cache} if refinement_cache is not None else {}),
                )
            finally:
                cleanup_started = time.perf_counter()
                release_warp_scene_memory(getattr(cfg, "warp_device", "cuda:0"))
                cleanup_s = time.perf_counter() - cleanup_started
            elapsed = time.perf_counter() - started
        record_service(elapsed, "physics_batch_calls")
        if len(values) != len(entries):
            raise PipelineError("mixed refinement returned a different number of candidates")
        info = dict(values[0].get("metrics", {}).get("mixed_batch", {})) if values else {}
        batches.append({
            "batch_id": len(batches), "world_capacity": capacity,
            "active_worlds": count, "padded_worlds": 0,
            "unique_scenes": len({record["bundle"]["scene_id"] for record, _, _, _ in entries}),
            "scene_epochs": sorted({(record["bundle"]["scene_id"], int(record["bundle"]["metadata"].get("epoch", 0))) for record, _, _, _ in entries}),
            "batch_service_s": elapsed, "graph_pool_cleanup_s": cleanup_s, **info,
        })
        for (record, offset, _name, _data), result in zip(entries, values):
            record["results"][offset] = result
        print(f"[shm:{stage}] mixed batch {len(batches)}: {count}/{capacity} worlds, "
              f"{batches[-1]['unique_scenes']} scenes", flush=True)
        accept_ready()

    for descriptor in descriptors:
        if cancel_event is not None and cancel_event.is_set():
            raise PipelineError("mixed world pipeline was cancelled")
        with cpu_lease():
            started = time.perf_counter()
            bundle = consume_scene_bundle(descriptor)
            scene_id = str(bundle["scene_id"])
            case = _shm_case(bundle, scene_id)
            candidates = [_shm_candidate(entry, scene_id) for entry in bundle.get("candidates", [])]
            record_service(time.perf_counter() - started, "input_deserialization")
        record = {"descriptor": descriptor, "bundle": bundle, "results": [None] * len(candidates)}
        records.append(record)
        if stage == "grasp":
            for offset, (relative, data) in enumerate(candidates):
                pending.append((record, offset, str(Path(logical_roots["init"]) / relative), data))
            while len(pending) >= capacity:
                run_refinement(capacity)
        elif candidates:
            with cpu_lease():
                started = time.perf_counter()
                _validate_shm_stability_scene(case)
                prepared, names = [], []
                for offset, (relative, data) in enumerate(candidates):
                    name = str(Path(logical_roots["stability_candidate"]) / relative)
                    if name in routes:
                        raise PipelineError(f"duplicate mixed stability input: {name}")
                    routes[name] = (record, offset)
                    item = dict(data)
                    item.update(scene_path=case["stability_scene"], pipeline_stage="stability_candidate",
                                source_refined_grasp_path=str(Path(logical_roots["grasp"]) / relative))
                    prepared.append(item)
                    names.append(name)
                report = stability_batcher.push(prepared, source_names=names)
                record_service(time.perf_counter() - started, "physics_batch_calls")
            accept_stability(report)
        accept_ready()
    if stage == "grasp" and pending:
        run_refinement(len(pending))
    if stability_batcher is not None:
        with cpu_lease():
            started = time.perf_counter()
            report = stability_batcher.flush()
            record_service(time.perf_counter() - started, "physics_batch_calls")
        accept_stability(report)
    accept_ready()
    if records or routes or pending:
        raise PipelineError("mixed world pipeline lost candidate results at end of stream")
    update_cache_stats()



def _preloaded_scene_paths(args, stage):
    """Use the frozen batch inventory, including scenes with no passing grasp."""
    if getattr(args, "warp_model_cache", "none") != "preloaded":
        return None
    key = {"grasp": "grasp_scene", "stability": "stability_scene"}[stage]
    cases = _load_index(Path(args.batch_index))["cases"]
    paths = [str(Path(case[key]).resolve()) for case in cases]
    if not paths or len(paths) != len(set(paths)) or any(not Path(path).is_file() for path in paths):
        raise PipelineError(f"invalid complete scene inventory for {stage} preloading")
    return paths



def _runtime_metadata() -> dict[str, Any]:
    import mujoco

    result = {"python": sys.executable, "mujoco": mujoco.__version__}
    for package in ("warp-lang", "mujoco-warp"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _stage_uses_warp(args, stage):
    return stage in {'grasp', 'stability'}


def _shared_logical_roots(args):
    grasp = Path(args.save_root) / f"{args.exp}_grasp_{args.hand}"
    stable = Path(args.save_root) / f"{args.exp}_stability_{args.hand}"
    return {'init':str(grasp/'init_data'), 'grasp':str(grasp/'grasp_data'),
            'stability_candidate':str(stable/'grasp_data'), 'stable':str(stable/'succ_grasp')}


def _grasp_overrides(args):
    return [f'op.pregrasp.cdist={args.pregrasp_cdist}',
            f'op.pregrasp.filter.collision.ho_thre={args.pregrasp_ho_thre}',
            '+refinement_backend=mujoco_warp_mixed', f'+warp_device={args.warp_device}',
            'physics_profile=native', f'+warp_world_batch_size={args.warp_world_batch_size}',
            '+warp_model_cache=preloaded']


def _process_shm_scene_impl(descriptor, *, bundle, precomputed_results):
    started = time.perf_counter()
    if _POOL_STAGE not in {'grasp','stability'}:
        raise PipelineError('Unsupported stage')
    scene_id = str(bundle['scene_id'])
    epoch = int(bundle.get('metadata',{}).get('epoch',0))
    case = _shm_case(bundle,scene_id)
    candidates = [_shm_candidate(e,scene_id) for e in bundle.get('candidates',[])]
    if len(precomputed_results) != len(candidates):
        raise PipelineError('Mixed physics returned a different candidate count')
    if _POOL_STAGE == 'stability' and candidates:
        _validate_shm_stability_scene(case)
    output, results, published = [], [], 0
    for (relative, source), stage_result in zip(candidates,precomputed_results):
        target = _SHM_ROOTS['grasp' if _POOL_STAGE=='grasp' else 'stable'] / relative
        ok = bool(stage_result.get('success'))
        metrics = stage_result.get('metrics')
        result = {'scene_id':scene_id,'success':ok,'path':str(target),**_result_metric_fields(metrics)}
        if isinstance(metrics,dict):
            if isinstance(metrics.get('timings'),dict): result['timings']=dict(metrics['timings'])
            if not ok and isinstance(metrics.get('failure_stage'),str): result['failure_stage']=metrics['failure_stage']
        if ok:
            data = stage_result.get('output_data')
            if not isinstance(data,dict): raise PipelineError('Successful candidate lacks output data')
            if _POOL_STAGE == 'grasp':
                output.append({'relative_path':str(relative),'data':data})
            else:
                published += int(_publish_template_feedback(scene_id,relative,source))
                data = dict(data)
                _stable_score(data,target)
                data['source_grasp_path'] = data['source_refined_grasp_path'] = None
                data['source_candidate_id'] = f'{scene_id}:{relative.as_posix()}'
                data['provenance'] = {'transport':'shared_memory','scene_id':scene_id,
                    'candidate_id':data['source_candidate_id'],'relative_path':relative.as_posix(),'final_stage':'stability'}
                target.parent.mkdir(parents=True,exist_ok=True)
                np.save(target,data)
        audit_event("candidate_result",stage=_POOL_STAGE,scene_id=scene_id,epoch=epoch,
                    relative=str(relative),success=ok,metrics=metrics,n_evo=source.get("n_evo"),
                    init_hard_all=source.get("audit_init_hard_all"))
        audit_sample(_POOL_STAGE,scene_id,ok,relative,source,stage_result)
        results.append(result)
    descriptor_out = None
    if _POOL_STAGE == 'grasp':
        descriptor_out = publish_scene_bundle(scene_id,output,namespace=_SHM_NAMESPACE,
                           metadata={'case':case,'stage':_POOL_STAGE,'epoch':epoch})
    return {'scene_id':scene_id,'epoch':epoch,'results':results,
        'models_built':sum(int(r.get('metrics',{}).get('model_builds',0)) for r in precomputed_results),
        'service_duration_s':time.perf_counter()-started,'input_size':int(descriptor['size']),
        'output_descriptor':descriptor_out,'templates_published':published}


def _shared_worker(args, input_queue=None, output_queue=None) -> None:
    """Stage entrypoint whose candidate data plane is POSIX shared memory."""

    stage_started = time.perf_counter()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    signal_dir = Path(args.signal_dir)
    grasp_exp = f"{args.exp}_grasp"
    stability_exp = f"{args.exp}_stability"
    logical_roots = _shared_logical_roots(args)
    cancel_event = getattr(args, "shm_cancel_event", None)
    cpu_slots = getattr(args, "cpu_slot_gate", None)
    template_feedback_queue = getattr(args, "template_feedback_queue", None)
    template_feedback_accepting_event = getattr(
        args, "template_feedback_accepting_event", None
    )
    template_max_evo = int(getattr(args, "template_max_evo", 2))
    template_max_buffer = int(getattr(args, "template_max_buffer", 1024))
    init_epochs = int(getattr(args, "init_epochs", 1))
    summary = _new_stage_summary(args.stage, "scene", "shared_memory")
    summary["runtime"] = _runtime_metadata()
    uses_warp = _stage_uses_warp(args, args.stage)
    if args.stage in {"grasp", "stability"}:
        summary["physics_profile"] = getattr(args, "physics_profile", None) or "upstream_mujoco333"
    summary["backend"] = (
        getattr(args, "refinement_backend", "mujoco")
        if args.stage == "grasp"
        else getattr(args, "stability_backend", "mujoco") if args.stage == "stability" else "mujoco"
    )
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [shm:{args.stage}] %(levelname)s %(message)s",
    )
    audit_event("stage_start",stage=args.stage,args={k:getattr(args,k,None) for k in ["warp_world_batch_size","init_epochs","final_grasps","seed","object_batch_size","cpu_slots","shm_namespace"]})
    print(f"[shm:{args.stage}] ready", flush=True)
    try:
        if args.stage == "init":
            from dexonomy.op.gen_init import operate_init

            cfg = _compose_cfg(
                _common_overrides(args, "init", grasp_exp)
                + [
                    f"tmpl_name={args.template}",
                    "init_gpu=[0]",
                    f"obj_mass={args.obj_mass}",
                    f"op.epoch={init_epochs}",
                    f"op.object.batch_size={args.object_batch_size}",
                    "op.object.n_cfg=0",
                    f"op.object.n_init_point={args.init_points}",
                    f"op.object.n_init_inplane={args.inplane}",
                    f"op.object.cfg_path={args.scene_glob}",
                    f"op.matcher.opt_step={args.opt_steps}",
                    f"op.filter.general.n_final={args.final_grasps}",
                    f"tmpl_max_evo={template_max_evo}",
                    f"tmpl_max_buf={template_max_buffer}",
                ]
            )
            cases = _case_map(_load_index(Path(args.batch_index)))
            published_scene_epochs: set[tuple[str, int]] = set()

            def publish_scene_epoch(
                scene_id: str,
                epoch: int,
                candidates: list[dict[str, Any]],
            ) -> None:
                audit_event("init_bundle",scene_id=scene_id,epoch=epoch,count=len(candidates),
                            candidates=[{"relative":x["relative_path"],"n_evo":x["data"].get("n_evo"),"hard_all":x["data"].get("audit_init_hard_all"),"origin":x["data"].get("audit_init_origin_index")} for x in candidates])
                case = cases.get(scene_id)
                if case is None:
                    raise PipelineError(f"initializer returned unknown scene: {scene_id}")
                key = (scene_id, int(epoch))
                if key in published_scene_epochs:
                    raise PipelineError(
                        "initializer published scene epoch more than once: "
                        f"{scene_id}, epoch {epoch}"
                    )
                descriptor = publish_scene_bundle(
                    scene_id,
                    candidates,
                    namespace=args.shm_namespace,
                    metadata={"case": case, "stage": "init", "epoch": int(epoch)},
                )
                for _ in candidates:
                    _record_stage_result(summary, scene_id, True)
                _record_epoch_bundle(
                    summary,
                    int(epoch),
                    [{"success": True} for _ in candidates],
                )
                summary["tasks"] += 1
                summary["scene_bundles_out"] += 1
                summary["shared_memory_bytes_out"] += int(descriptor["size"])
                _shm_queue_put(output_queue, descriptor, cancel_event)
                published_scene_epochs.add(key)

            init_result = operate_init(
                cfg,
                scene_epoch_data_callback=publish_scene_epoch,
                persist_results=False,
                template_feedback_queue=template_feedback_queue,
                template_feedback_accepting_event=(
                    template_feedback_accepting_event
                ),
                template_feedback_cancel_event=cancel_event,
            )
            summary["template_feedback"] = init_result.get("template_feedback")
            expected_scene_epochs = {
                (scene_id, epoch)
                for scene_id in cases
                for epoch in range(init_epochs)
            }
            missing_scene_epochs = expected_scene_epochs.difference(
                published_scene_epochs
            )
            if missing_scene_epochs:
                raise PipelineError(
                    "initializer did not publish every batch scene epoch: "
                    f"{sorted(missing_scene_epochs)}"
                )
            if cpu_slots is not None:
                cpu_slots.set_capacity(int(args.cpu_slot_target))
                cpu_slots.set_stage_targets(args.cpu_phase_targets["after_init"])
        elif args.stage == "grasp":
            cfg = _compose_cfg(
                _common_overrides(args, "grasp", grasp_exp)
                + [
                    "tmpl_upd_mode=hybrid",
                    f"tmpl_max_evo={template_max_evo}",
                ]
                + _grasp_overrides(args)
            )
            processor = _process_shm_mixed_descriptors
            processor(
                _consume_shm_queue(input_queue, cancel_event),
                stage="grasp",
                cfg=cfg,
                workers=args.workers,
                pool_start_method="spawn" if uses_warp else "fork",
                summary=summary,
                namespace=args.shm_namespace,
                logical_roots=logical_roots,
                output_queue=output_queue,
                cancel_event=cancel_event,
                cpu_slots=cpu_slots,
                **({"scene_paths": _preloaded_scene_paths(args, "grasp")}
                   if getattr(args, "warp_model_cache", "none") == "preloaded" else {}),
            )
            if cpu_slots is not None:
                cpu_slots.set_stage_targets(args.cpu_phase_targets["after_grasp"])
        elif args.stage == "stability":
            cfg = _compose_cfg(
                _common_overrides(args, "eval", stability_exp)
                + [
                    f"obj_mass={args.obj_mass}",
                    "op.reuse_fc_preamble="
                    f"{str(args.force_closure_snapshot == 'on').lower()}",
                ]
                + ([f"+stability_backend={args.stability_backend}", f"+warp_device={args.warp_device}"]
                   if uses_warp else [])
                + ([f"physics_profile={args.physics_profile}"]
                   if getattr(args, "physics_profile", None) is not None else [])
                + ([f"+warp_world_batch_size={args.warp_world_batch_size}",
                    f"+warp_model_cache={getattr(args, 'warp_model_cache', 'none')}"]
                   if args.stability_backend == "mujoco_warp_mixed" else [])
                + ["+terminal_stability=True"]
            )
            processor = _process_shm_mixed_descriptors
            processor(
                _consume_shm_queue(input_queue, cancel_event),
                stage="stability",
                cfg=cfg,
                workers=3 * args.workers,
                pool_start_method="spawn" if uses_warp else "fork",
                summary=summary,
                namespace=args.shm_namespace,
                logical_roots=logical_roots,
                output_queue=output_queue,
                cancel_event=cancel_event,
                cpu_slots=cpu_slots,
                template_feedback_queue=template_feedback_queue,
                template_feedback_accepting_event=(
                    template_feedback_accepting_event
                ),
                template_feedback_max_evo=template_max_evo,
                **({"scene_paths": _preloaded_scene_paths(args, "stability")}
                   if getattr(args, "warp_model_cache", "none") == "preloaded" else {}),
            )
            if cpu_slots is not None:
                cpu_slots.set_stage_targets(
                    args.cpu_phase_targets["after_stability"]
                )
        else:
            raise PipelineError(f"unknown shared-memory stage: {args.stage}")
        summary["wall_duration_s"] = time.perf_counter() - stage_started
        _write_stage_summary(signal_dir, summary)
        _signal(signal_dir / f"{args.stage}.done", args.stage, summary["processed"])
        print(
            f"[shm:{args.stage}] complete ({summary['processed']} processed, "
            f"{summary['success']} success)",
            flush=True,
        )
    except BaseException:
        if cancel_event is not None:
            cancel_event.set()
        raise
    finally:
        if (
            output_queue is not None
            and (cancel_event is None or not cancel_event.is_set())
        ):
            _shm_queue_put(output_queue, None, cancel_event)



def _run_shared_memory(
    args, index: dict[str, Any], signal_dir: Path
) -> dict[str, Any]:
    if getattr(args, "task_unit", "scene") != "scene":
        raise PipelineError("shared-memory transport requires --task-unit scene")
    ctx = multiprocessing.get_context("spawn")
    detected_cpu_slots = detect_available_cpu_slots()
    cpu_scheduling = getattr(args, "cpu_scheduling", "static")
    requested_cpu_slots = int(getattr(args, "cpu_slots", 0))
    cpu_init_reserve = int(getattr(args, "cpu_init_reserve", 2))
    target_cpu_slots = detected_cpu_slots
    def split_targets(
        capacity: int,
        *,
        grasp_target: int,
    ) -> dict[str, int]:
        grasp = min(max(0, grasp_target), capacity)
        return {
            "grasp": grasp,
            "stability": capacity - grasp,
        }

    cpu_slot_gate = None
    initial_cpu_slots = target_cpu_slots
    cpu_phase_targets = None
    if cpu_scheduling == "dynamic":
        if requested_cpu_slots < 0:
            raise PipelineError(
                "--cpu-slots must be zero (auto) or a positive integer"
            )
        if cpu_init_reserve < 0:
            raise PipelineError("--cpu-init-reserve must be non-negative")
        target_cpu_slots = requested_cpu_slots or detected_cpu_slots
        if target_cpu_slots > detected_cpu_slots:
            raise PipelineError(
                f"requested {target_cpu_slots} CPU slots but the cgroup/affinity "
                f"capacity is {detected_cpu_slots}"
            )
        if cpu_init_reserve >= target_cpu_slots:
            raise PipelineError(
                "--cpu-init-reserve must be smaller than the effective CPU slots"
            )
        initial_cpu_slots = max(1, target_cpu_slots - cpu_init_reserve)
        cpu_phase_targets = {
            "during_init": split_targets(
                initial_cpu_slots,
                grasp_target=args.workers,
            ),
            "after_init": split_targets(
                target_cpu_slots,
                grasp_target=args.workers,
            ),
            "after_grasp": split_targets(
                target_cpu_slots,
                grasp_target=0,
            ),
            "after_stability": split_targets(
                target_cpu_slots,
                grasp_target=0,
            ),
        }
        cpu_slot_gate = PriorityCpuSlots(
            initial_cpu_slots,
            context=ctx,
            stage_targets=cpu_phase_targets["during_init"],
        )
    elif cpu_scheduling != "static":
        raise PipelineError(f"unsupported CPU scheduling mode: {cpu_scheduling}")

    args.detected_cpu_slots = detected_cpu_slots
    args.cpu_slot_target = target_cpu_slots
    args.cpu_slot_initial = initial_cpu_slots
    args.cpu_slot_gate = cpu_slot_gate
    args.cpu_phase_targets = cpu_phase_targets
    template_feedback_enabled = getattr(args, "template_feedback", "off") == "on"
    feedback_queue = None
    feedback_accepting_event = None
    if template_feedback_enabled:
        feedback_depth = max(1, int(args.template_feedback_queue_depth))
        feedback_queue = ctx.Queue(maxsize=feedback_depth)
        feedback_accepting_event = ctx.Event()
        feedback_accepting_event.set()
    args.template_feedback_queue = feedback_queue
    args.template_feedback_accepting_event = feedback_accepting_event
    queue_depth = max(1, int(args.shared_memory_queue_depth))
    init_to_grasp = ctx.Queue(maxsize=queue_depth)
    grasp_to_stability = ctx.Queue(maxsize=queue_depth)
    args.shm_namespace = f"p{os.getpid()}_{uuid.uuid4().hex[:12]}"
    args.shm_cancel_event = ctx.Event()
    stage_queues = {
        "stability": (grasp_to_stability, None),
        "grasp": (init_to_grasp, grasp_to_stability),
    }
    processes = []
    for stage in ("stability", "grasp"):
        stage_args = argparse.Namespace(**vars(args))
        stage_args.stage = stage
        input_queue, output_queue = stage_queues[stage]
        process = ctx.Process(
            target=_shared_worker,
            args=(stage_args, input_queue, output_queue),
            name=f"dexonomy-shm-{stage}",
        )
        process.start()
        processes.append((stage, process))

    pending = dict(processes)
    monitor_stop = threading.Event()
    feedback_drain_stop = threading.Event()
    feedback_drainer = None

    def drain_closed_feedback() -> None:
        while not feedback_drain_stop.is_set():
            try:
                feedback_queue.get(timeout=0.1)
            except queue.Empty:
                continue

    def monitor_children() -> None:
        while not monitor_stop.wait(0.2):
            for _stage, process in processes:
                code = process.exitcode
                if code is not None and code != 0:
                    args.shm_cancel_event.set()
                    return

    monitor = threading.Thread(target=monitor_children, daemon=True)
    monitor.start()
    run_failed = False
    try:
        # As in the path-queue baseline, keep CUDA/Warp initialization in the
        # coordinator rather than spawning it after the CPU stages.
        init_args = argparse.Namespace(**vars(args))
        init_args.stage = "init"
        _shared_worker(init_args, None, init_to_grasp)
        if feedback_queue is not None:
            # Init is the sole template consumer.  Once it closes acceptance,
            # drain any producer race at the boundary so multiprocessing queue
            # feeder threads can terminate cleanly.
            feedback_drainer = threading.Thread(
                target=drain_closed_feedback, daemon=True
            )
            feedback_drainer.start()
        if args.shm_cancel_event.is_set():
            raise PipelineError("a shared-memory stage failed during initialization")
        if cpu_slot_gate is not None:
            cpu_slot_gate.set_capacity(target_cpu_slots)
        while pending:
            if args.shm_cancel_event.is_set():
                raise PipelineError("a shared-memory stage cancelled the pipeline")
            for stage, process in list(pending.items()):
                if process.is_alive():
                    continue
                process.join()
                pending.pop(stage)
                if process.exitcode != 0:
                    raise PipelineError(
                        f"shared-memory stage failed: {stage} ({process.exitcode})"
                    )
            if pending and args.shm_cancel_event.wait(0.2):
                raise PipelineError("a shared-memory stage cancelled the pipeline")
    except BaseException:
        run_failed = True
        args.shm_cancel_event.set()
        for process in pending.values():
            process.terminate()
        for process in pending.values():
            process.join(timeout=10)
        for process in pending.values():
            if process.is_alive():
                process.kill()
        for process in pending.values():
            process.join(timeout=10)
        raise
    finally:
        monitor_stop.set()
        monitor.join(timeout=2)
        feedback_drain_stop.set()
        if feedback_drainer is not None:
            feedback_drainer.join(timeout=2)
        for scene_queue in (
            init_to_grasp,
            grasp_to_stability,
        ):
            if scene_queue is None:
                continue
            if run_failed:
                scene_queue.cancel_join_thread()
            scene_queue.close()
            if not run_failed:
                scene_queue.join_thread()
        if feedback_queue is not None:
            if run_failed:
                feedback_queue.cancel_join_thread()
            feedback_queue.close()
            if not run_failed:
                feedback_queue.join_thread()
        removed = cleanup_scene_shm_namespace(args.shm_namespace)
        if removed:
            logging.warning(
                "cleaned %d unconsumed shared-memory scene bundles", len(removed)
            )

    return _finalize_online_run(
        args, index, signal_dir, transport="shared_memory"
    )



def _finalize_online_run(args, index, signal_dir, *, transport):
    summaries = {}
    for stage in ('init', 'grasp', 'stability'):
        path = signal_dir / f'{stage}.summary.json'
        if not path.is_file() or not (signal_dir / f'{stage}.done').is_file():
            raise PipelineError(f'Incomplete stage: {stage}')
        summaries[stage] = json.loads(path.read_text())
    output = Path(args.save_root) / f'{args.exp}_stability_{args.hand}'
    output.mkdir(parents=True, exist_ok=True)
    report = build_batch_report_from_summaries(index['batch_index'], summaries, output / 'batch_report.json')
    learning = {'enabled':True, 'mode':'original_dexsyn_hybrid_online_feedback',
        'max_evolution':args.template_max_evo, 'max_buffer':args.template_max_buffer,
        'queue_depth':args.template_feedback_queue_depth,'epochs':args.init_epochs,
        'attempts_per_object_per_epoch':args.final_grasps,
        'consumer':summaries['init'].get('template_feedback'),
        'published_by_stability':summaries['stability']['template_feedback_published']}
    total_seconds = time.perf_counter() - args.pipeline_started
    summary = {'format_version':'dexonomyopt-pipeline-summary/v1', 'version':'1.3.1',
        'final_stage':'stability', 'transport':transport, 'stages':summaries,
        'totals':report['totals'], 'success_count':report['success_count'],
        'gsr':report['gsr'], 'osr':report['osr'], 'template_learning':learning,
        'total_duration_s':total_seconds,
        'stable_grasps_per_min':60*report['totals']['stable_grasps']/total_seconds,
        'cpu_scheduler':{'detected_slots':args.detected_cpu_slots,
            'target_slots':args.cpu_slot_target,'phase_targets':args.cpu_phase_targets,
            'slot_statistics':args.cpu_slot_gate.snapshot() if args.cpu_slot_gate else None}}
    (output/'pipeline_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({key:summary[key] for key in ('version','final_stage','totals','gsr','osr','total_duration_s')},indent=2),flush=True)
    return report


def run(args):
    args.pipeline_started = time.perf_counter()
    signal_dir = Path(args.save_root) / f'.{args.exp}_signals'
    roots = _shared_logical_roots(args)
    if signal_dir.exists() or any(Path(p).parent.exists() for p in roots.values()):
        raise PipelineError('Experiment output already exists; use a new --exp (no implicit overwrite)')
    converted = Path(args.converted_root)
    if args.reuse_converted:
        index = _load_index(converted/'batch_index.json')
        if Path(index['manifest']).resolve() != Path(args.manifest).resolve():
            raise PipelineError('Converted batch belongs to a different manifest')
        index['batch_index'] = str(converted/'batch_index.json')
    else:
        index = convert_batch(args.manifest, converted)
    profiles = [case['runtime_profile'] for case in index['cases']]
    masses = [float(p['target_mass_kg']) for p in profiles]
    if not masses or not all(np.isfinite(m) and m>0 for m in masses) or max(masses)-min(masses)>1e-9:
        raise PipelineError('All objects in a run must have the same positive target mass')
    signal_dir.mkdir(parents=True)
    args.batch_index = index['batch_index']
    args.scene_glob = index['grasp_scene_glob']
    args.signal_dir = str(signal_dir)
    args.obj_mass = masses[0]
    config = {k:v for k,v in vars(args).items() if k!='pipeline_started'}
    (signal_dir/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
    _run_shared_memory(args,index,signal_dir)
