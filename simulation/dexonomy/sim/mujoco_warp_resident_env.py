"""GPU-resident, same-scene batches for the native Dexonomy refinement loop.

The host submits a fixed maximum schedule; no device value is read inside either
control loop. Conditional contact queries restore the unqueried worlds' derived
state. Completed worlds retain their first stopping state, not their last padded
simulation state. CPU QP runs only at the grasp/pregrasp phase boundary.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict

import numpy as np
import mujoco_warp as mjw
from mujoco_warp._src.types import ContactType
import warp as wp

wp.set_module_options({"enable_backward": False})

from . import warp_refinement_kernels as k
from .upstream_physics import needs_legacy_forward_warmstart
from .mujoco_warp_env import (
    preserve_model_options, check_warp_allocation, MuJoCoWarpAllocationError,
    mixed_world_buffer_sizes,
)


@wp.kernel
def local_normals(bodies: wp.array2d[int], counts: wp.array[int], normals: wp.array2d[wp.vec3],
                  rotations: wp.array2d[wp.mat33], output: wp.array2d[wp.vec3]):
    w, i = wp.tid()
    if i < counts[w]:
        output[w, i] = wp.transpose(rotations[w, bodies[w, i]]) @ normals[w, i]


@wp.kernel
def invert_mask(active: wp.array[int], inactive: wp.array[int]):
    w = wp.tid()
    inactive[w] = 1 - active[w]


@wp.kernel
def mask_summary(active: wp.array[int], query: wp.array[int], total: wp.array[int],
                 queried: wp.array[int], inverse: wp.array[int]):
    n = int(0)
    nq = int(0)
    for w in range(active.shape[0]):
        n += active[w]
        nq += query[w]
        inverse[w] = 1 - query[w]
    total[0] = n
    queried[0] = nq


@wp.kernel
def remember_overflow(flags: wp.array[int], nacon: wp.array[int], nefc: wp.array[int],
                      contact_capacity: int, constraint_capacity: int,
                      solver_mask: int, history: wp.array[int]):
    w = wp.tid()
    value = flags[w] & ~solver_mask
    if nacon[0] > contact_capacity or nefc[w] > constraint_capacity:
        value = value | 1073741824
    history[w] = history[w] | value


@wp.kernel
def remember_nonfinite(active: wp.array[int], qpos: wp.array2d[float],
                       qvel: wp.array2d[float], history: wp.array[int]):
    w = wp.tid()
    if active[w] != 0:
        bad = int(0)
        for j in range(qpos.shape[1]):
            if not wp.isfinite(qpos[w, j]):
                bad = 536870912
        for j in range(qvel.shape[1]):
            if not wp.isfinite(qvel[w, j]):
                bad = 536870912
        history[w] = history[w] | bad


@wp.kernel
def point_wrenches(active: wp.array[int], bodies: wp.array2d[int], counts: wp.array[int],
                   points: wp.array2d[wp.vec3], forces: wp.array2d[wp.vec3],
                   xipos: wp.array2d[wp.vec3], out: wp.array2d[wp.spatial_vector]):
    w, b = wp.tid()
    f = wp.vec3(0.0)
    t = wp.vec3(0.0)
    if active[w] != 0:
        for i in range(counts[w]):
            if bodies[w, i] == b:
                f += forces[w, i]
                t += wp.cross(points[w, i] - xipos[w, b], forces[w, i])
    out[w, b] = wp.spatial_vector(f, t)


@wp.kernel
def squeeze_pose(qpos: wp.array2d[float], qfrc: wp.array2d[float],
                 divisor: wp.array[float], out: wp.array2d[float]):
    w, j = wp.tid()
    value = qpos[w, j]
    if j >= 7:
        value += qfrc[w, j - 1] / divisor[j - 1]
    out[w, j] = value


class ResidentRefinementBatch:
    # State needed to resume integration and preserve the original controller's
    # use of mj_step's pre-integration derived quantities. Other forward outputs
    # (mass/constraint/contact workspaces) are rebuilt by the next mjw.step.
    STATE = ("time", "qpos", "qvel", "act", "history", "qacc_warmstart", "ctrl",
             "qfrc_applied", "xfrc_applied", "eq_active", "mocap_pos", "mocap_quat",
             "userdata", "qacc", "act_dot", "xpos", "xquat", "xmat", "xipos",
             "ximat", "cdof", "subtree_com")
    QUERY_DERIVED = ("history", "qacc_warmstart", "qacc", "act_dot", "xpos", "xquat", "xmat",
                     "xipos", "ximat", "cdof", "subtree_com")

    def __init__(self, host, candidates, cfg, allocation_scale=1, mixed=None):
        self.host, self.cfg, self.candidates = host, cfg, candidates
        self.mixed = mixed
        self.world_hosts = ([mixed.hosts[int(i)] for i in mixed.world_host_indices]
                            if mixed is not None else [host] * len(candidates))
        self.nworld = len(candidates)
        self.device = str(cfg.get("warp_device", "cuda:0"))
        self._legacy_forward_warmstart = needs_legacy_forward_warmstart(host._cfg.physics_profile)
        wp.set_device(self.device)
        self.profile = os.environ.get("DEXONOMY_RESIDENT_PROFILE", "0") == "1"
        self.events = []
        self.timings = defaultdict(float)
        self.m = mixed.warp_model if mixed is not None else mjw.put_model(host._model)
        preserve_model_options(host._model, self.m)
        if host._model.nmocap != 1:
            raise ValueError("resident refinement requires the native free hand with one wrist mocap")
        if int(host._model.opt.enableflags) & int(mjw.EnableBit.SLEEP):
            raise ValueError("resident refinement does not support sleeping state")
        self.hand_nq = host._model.nq - len(host._obj_init_qpos)
        allocation_model = mixed.carrier_model if mixed is not None else host._model
        if mixed is None:
            allocation = {"nconmax": allocation_scale * max(512, allocation_model.ngeom * 12),
                          "njmax": allocation_scale * max(2048, host._model.nv * 32)}
        else:
            allocation = mixed_world_buffer_sizes(host._model.nv, allocation_scale, refinement=True)
            ccd_multiplier=int(cfg.get("refine_ccd_multiplier",1))
            if ccd_multiplier<1:raise ValueError("refine_ccd_multiplier must be positive")
            allocation["nccdmax"] *= ccd_multiplier
            from dexonomy.night_contracts import small_batch_allocation
            allocation=small_batch_allocation(allocation,self.nworld,int(cfg.get("refine_small_batch_limit",0)),int(cfg.get("refine_small_batch_capacity",0)))
        from dexonomy.night_contracts import constraint_allocation, contact_allocation
        allocation=constraint_allocation(allocation,cfg.get("refine_njmax",0))
        if mixed is not None:allocation=contact_allocation(allocation,cfg.get("refine_nconmax",0),cfg.get("refine_nccdmax",0))
        from dexonomy.sim.data_workspace import acquire
        self._memory_audit('before_data')
        self.d = acquire(allocation_model, self.m, nworld=self.nworld, **allocation)
        self._memory_audit('after_data')
        if mixed is not None:
            mixed.metadata["allocation"] = {**allocation,
                "nworld": self.nworld, "naconmax": int(self.d.naconmax),
                "naccdmax": int(self.d.naccdmax), "njmax_nnz": int(self.d.njmax_nnz)}
        if mixed is None:
            self.m.geom_margin = wp.array(np.repeat(host._model.geom_margin[None], self.nworld, 0), dtype=float)
            object_mask = np.array([int(host._model.geom(i).name.startswith(host._cfg.obj_prefix))
                                    for i in range(host._model.ngeom)], dtype=np.int32)
        else:
            mixed.initialize_data(self.d)
            if self.m.geom_margin.shape[0] == 1:
                self.m.geom_margin = wp.array(np.repeat(self.m.geom_margin.numpy(), self.nworld, 0), dtype=float)
            object_mask = (allocation_model.geom_bodyid >= host._obj_active_bodyid).astype(np.int32)
        self.object_geoms = wp.array(object_mask, dtype=int)
        self.ctrl_map = wp.array(host._hand_qpos2ctrl_mat[:, 6:].astype(np.float32), dtype=float)
        self.active = wp.ones(self.nworld, dtype=int)
        self.query = wp.zeros(self.nworld, dtype=int)
        self.inverse_query = wp.ones(self.nworld, dtype=int)
        self.inactive = wp.zeros(self.nworld, dtype=int)
        self.newly_done = wp.zeros(self.nworld, dtype=int)
        self.total_active = wp.full(1, self.nworld, dtype=int)
        self.total_query = wp.zeros(1, dtype=int)
        self.iterations = wp.zeros(self.nworld, dtype=int)
        self.error_history = wp.zeros(self.nworld, dtype=int)
        self.hh = wp.zeros(self.nworld, dtype=float)
        self.ho = wp.zeros(self.nworld, dtype=float)
        self.margins = wp.zeros(self.nworld, dtype=float)
        self.saved = {name: wp.empty_like(getattr(self.d, name)) for name in self.STATE}
        self.query_saved = {name: wp.empty_like(getattr(self.d, name)) for name in self.QUERY_DERIVED}
        self.saved_hh = wp.empty_like(self.hh)
        self.saved_ho = wp.empty_like(self.ho)
        self.graphs = {}
        self._memory_audit('after_snapshots')
        self._initialize()
        self._memory_audit('after_initialize')

    def _memory_audit(self, phase):
        if self.cfg.get('capacity_memory_audit', False):
            import torch
            from dexonomy.acceptance_audit import event
            free,total=torch.cuda.mem_get_info()
            event('refine_memory',phase=phase,nworld=self.nworld,used_bytes=total-free,free_bytes=free)

    def _copy(self, mapping, *, to_saved, mask=None):
        for name, snapshot in mapping.items():
            current = getattr(self.d, name)
            source, target = (current, snapshot) if to_saved else (snapshot, current)
            if not source.size:
                continue
            if mask is None:
                wp.copy(target, source)
            else:
                k.copy_world_array(mask, source, target)

    def _margin(self, value):
        wp.launch(k.fill_world_margins, self.nworld, inputs=[float(value), self.margins])
        self._set_margin_array()

    def _set_margin_array(self):
        wp.launch(k.set_object_margin, (self.nworld, self.m.ngeom),
                  inputs=[self.object_geoms, self.margins, self.m.geom_margin])

    def _errors(self):
        wp.launch(remember_overflow, self.nworld, inputs=[self.d.overflow, self.d.nacon,
            self.d.nefc, self.d.naconmax, self.d.njmax,
            int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS), self.error_history])
        wp.launch(remember_nonfinite, self.nworld, inputs=[self.active, self.d.qpos,
                  self.d.qvel, self.error_history])

    def _check(self):
        wp.synchronize_device(self.device)
        flags = self.error_history.numpy()
        if np.any(flags):
            if np.any(flags & 536870912):
                from dexonomy.op.simulation_isolation import RefinementNonfiniteError
                # Keep mixed overflow bits: the isolation classifier rejects them.
                fields={}
                for name in ('qpos','qvel'):
                    values=getattr(self.d,name).numpy()
                    fields[name]=np.argwhere(~np.isfinite(values)).tolist()
                raise RefinementNonfiniteError(dict(history_flags=flags.tolist(),nonfinite_fields=fields))
            diagnostics = {"history_flags": flags.tolist(), "nacon": int(self.d.nacon.numpy()[0]),
                           "nacon_capacity": self.d.naconmax, "nefc": self.d.nefc.numpy().tolist(),
                           "nefc_capacity": self.d.njmax}
            raise MuJoCoWarpAllocationError(
                f"resident refinement allocation overflow: {diagnostics}", diagnostics)
        check_warp_allocation(self.d, mjw.OverflowType, context="resident refinement")

    def _capture(self, name, function):
        started = time.perf_counter()
        with wp.ScopedCapture() as capture:
            function()
        self.graphs[name] = capture.graph
        self._memory_audit('capture_'+name)
        self.timings["graph_capture_s"] += time.perf_counter() - started

    def _enqueue(self, label, function):
        if self.profile:
            start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
            wp.record_event(start)
            function()
            wp.record_event(end)
            self.events.append((label, start, end))
        else:
            function()

    def _launch(self, name, label=None):
        self._enqueue(label or name, lambda: wp.capture_launch(self.graphs[name]))

    def _summary(self):
        wp.launch(mask_summary, 1, inputs=[self.active, self.query, self.total_active,
                                         self.total_query, self.inverse_query])

    def _controls(self):
        wp.launch(k.prepare_controls, self.nworld, inputs=[self.active, self.d.qpos,
            self.ctrl_map, self.d.ctrl, self.d.qvel, self.d.qfrc_applied, self.d.xfrc_applied])

    def _forward(self):
        mjw.forward(self.m, self.d)
        if self._legacy_forward_warmstart:
            # Upstream pins MuJoCo 3.3.3: standalone forward updated warmstart.
            # This device copy is captured with the graph; no host readback.
            wp.copy(self.d.qacc_warmstart, self.d.qacc)

    def _reduce_contacts(self, mask):
        c = self.d.contact
        wp.launch(k.initialize_contact_minima, self.nworld, inputs=[self.hh, self.ho])
        wp.launch(k.reduce_contact_distances_parallel, self.d.naconmax, inputs=[mask, self.d.nacon,
            c.worldid, c.geom, c.dist, c.type, int(ContactType.CONSTRAINT | ContactType.PASSIVE),
            self.m.geom_bodyid, self.host.n_hand_body, self.m.nmocap, self.hh, self.ho])

    def _initialize(self):
        qpos = np.stack([np.concatenate((x["grasp_qpos"][0], h._obj_init_qpos))
                         for x, h in zip(self.candidates, self.world_hosts)]).astype(np.float32)
        self.d.qpos.assign(qpos)
        self.d.mocap_pos.assign(qpos[:, None, :3])
        # MuJoCo Warp quaternions preserve native wxyz storage.
        self.d.mocap_quat.assign(qpos[:, None, 3:7])
        self.d.qvel.zero_()
        self.d.ctrl.assign(qpos[:, 7:self.hand_nq] @ self.host._hand_qpos2ctrl_mat[:, 6:].T)
        self._margin(float(self.cfg.op.pregrasp.cdist))
        self._forward()
        self._memory_audit('after_initial_forward')
        self._errors()
        self._check()
        counts = np.array([len(x["hand_cbody"]) for x in self.candidates], dtype=np.int32)
        if np.any(counts == 0):
            raise ValueError("resident refinement requires at least one template contact")
        self.width = int(max(counts))
        ids = np.zeros((self.nworld, self.width), np.int32)
        world_points = np.zeros((self.nworld, self.width, 3), np.float32)
        targets, normals = np.zeros_like(world_points), np.zeros_like(world_points)
        hand_normals = np.zeros_like(world_points)
        for w, candidate in enumerate(self.candidates):
            count = counts[w]
            ids[w, :count] = [self.host._model.body(self.host._cfg.hand_prefix + name).id
                              for name in candidate["hand_cbody"]]
            world_points[w, :count] = candidate["hand_cpn_w"][:, :3]
            hand_normals[w, :count] = candidate["hand_cpn_w"][:, 3:]
            targets[w, :count] = candidate["obj_cpn_w"][:, :3]
            normals[w, :count] = candidate["obj_cpn_w"][:, 3:]
        self.bodies = wp.array(ids, dtype=int)
        self.counts = wp.array(counts, dtype=int)
        self.local = wp.zeros((self.nworld, self.width), dtype=wp.vec3)
        self.targets = wp.array(targets, dtype=wp.vec3)
        self.normals = wp.array(normals, dtype=wp.vec3)
        self.current_dist = wp.zeros((self.nworld, self.width), dtype=float)
        self.previous_dist = wp.zeros_like(self.current_dist)
        wp.launch(k.initialize_local_points, (self.nworld, self.width), inputs=[self.bodies,
            self.counts, wp.array(world_points, dtype=wp.vec3), self.d.xpos, self.d.xmat, self.local])
        normals_local = wp.zeros_like(self.local)
        wp.launch(local_normals, (self.nworld, self.width), inputs=[self.bodies, self.counts,
            wp.array(hand_normals, dtype=wp.vec3), self.d.xmat, normals_local])
        self.local_host = self.local.numpy()  # one setup-boundary copy, reused for template update
        self.local_normals_host = normals_local.numpy()
        self._margin(float(self.cfg.op.grasp.cdist))
        self._copy(self.saved, to_saved=True)

    def refine(self):
        cfg = self.cfg.op.grasp
        def force():
            self._controls()
            wp.launch(k.spring_body_wrenches, (self.nworld, self.m.nbody), inputs=[self.active,
                self.bodies, self.counts, self.local, self.targets, self.normals,
                self.d.xpos, self.d.xmat, self.d.xipos, self.current_dist, self.d.xfrc_applied])
            self.d.qfrc_applied.zero_()
            mjw.xfrc_accumulate(self.m, self.d, self.d.qfrc_applied)
            self.d.xfrc_applied.zero_()
        def physics():
            # Stock MJWarp cannot skip individual worlds. Restart padded worlds
            # from their saved stopping state rather than letting them drift.
            wp.launch(invert_mask, self.nworld, inputs=[self.active, self.inactive])
            self._copy(self.saved, to_saved=False, mask=self.inactive)
            for _ in range(int(cfg.substep)):
                mjw.step(self.m, self.d)
                self._errors()
        def query_forward():
            self._copy(self.query_saved, to_saved=True)
            self._margin(float(cfg.filter.contact.thre))
            self._forward()
            self._errors()
            self._reduce_contacts(self.query)
            self._copy(self.query_saved, to_saved=False, mask=self.inverse_query)
            self._margin(float(cfg.cdist))
        self._capture("grasp_force", lambda: wp.capture_if(self.total_active, on_true=force))
        # MJWarp allocates solver scratch while capturing. CUDA forbids those
        # allocation nodes inside an outer conditional body. Submit a fixed
        # schedule and mask/restore worlds instead; internal solver conditionals
        # still execute on device, and no CPU readback is introduced.
        self._capture("physics", physics)
        self._capture("contact_query", query_forward)
        self._capture("snapshot", lambda: self._copy(self.saved, to_saved=True, mask=self.newly_done))
        started = time.perf_counter()
        for iteration in range(int(cfg.step)):
            self._summary()
            self._launch("grasp_force")
            self._launch("physics", "grasp_physics")
            self._enqueue("stop_check", lambda: wp.launch(k.update_progress, self.nworld,
                inputs=[self.active, self.counts, self.current_dist, self.previous_dist, iteration, self.query]))
            self._summary()
            self._launch("contact_query")
            self._enqueue("stop_check", lambda: wp.launch(k.mark_grasp_done, self.nworld,
                inputs=[self.active, self.query, self.hh, self.ho, float(cfg.filter.collision.hh_thre),
                    float(cfg.filter.collision.ho_thre), iteration, int(cfg.step), self.newly_done, self.iterations]))
            self._launch("snapshot")
        self._check()
        self.timings["grasp_loop_wall_s"] = time.perf_counter() - started
        self.grasp_iterations = self.iterations.numpy()
        self._copy(self.saved, to_saved=False)
        self._margin(float(cfg.filter.contact.thre))
        self._capture("final_contact_forward", lambda: (self._forward(), self._errors()))
        started = time.perf_counter()
        self._launch("final_contact_forward")
        self._check()
        self._margin(float(cfg.cdist))
        self.timings["final_contact_forward_wall_s"] = time.perf_counter() - started
        return self.export_grasps()

    def export_grasps(self):
        started = time.perf_counter()
        d, c = self.d, self.d.contact
        batch = {name: getattr(d, name).numpy() for name in ("qpos", "xpos", "xmat")}
        n = int(d.nacon.numpy()[0])
        contacts = {name: getattr(c, name).numpy()[:n] for name in ("worldid", "geom", "dist", "pos", "frame", "type")}
        batch["contacts"] = contacts
        self.timings["grasp_download_s"] = time.perf_counter() - started
        return batch

    def pregrasp(self, passing, contact_sets):
        cfg = self.cfg.op.pregrasp
        self.active.assign(np.asarray(passing, dtype=np.int32))
        self.query.zero_()
        self.iterations.zero_()
        # QP output is the only phase-boundary upload. No per-iteration downloads.
        width = max(1, max((len(x["bn1"]) if x is not None else 0) for x in contact_sets))
        bodies = np.zeros((self.nworld, width), np.int32)
        points = np.zeros((self.nworld, width, 3), np.float32)
        forces = np.zeros_like(points)
        counts = np.zeros(self.nworld, np.int32)
        for w, contacts in enumerate(contact_sets):
            if contacts is not None and passing[w]:
                counts[w] = len(contacts["bn1"])
                bodies[w, :counts[w]] = [self.host._model.body(self.host._cfg.hand_prefix + name).id
                                        for name in contacts["bn1"]]
                points[w, :counts[w]] = contacts["pos"]
                forces[w, :counts[w]] = contacts["wrench"][:, :3]
        self.d.qfrc_applied.zero_()
        wp.launch(point_wrenches, (self.nworld, self.m.nbody), inputs=[self.active,
            wp.array(bodies, dtype=int), wp.array(counts, dtype=int), wp.array(points, dtype=wp.vec3),
            wp.array(forces, dtype=wp.vec3), self.d.xipos, self.d.xfrc_applied])
        mjw.xfrc_accumulate(self.m, self.d, self.d.qfrc_applied)
        self.d.xfrc_applied.zero_()
        divisor = np.ones(self.host._hand_nv, dtype=np.float32)
        for i in range(self.host._hand_nv):
            ids = np.flatnonzero(self.host._hand_qpos2ctrl_mat[:, i])
            if len(ids):
                divisor[i] = self.host._model.actuator_gainprm[ids[0], 0] * self.host._hand_qpos2ctrl_mat[ids[0]].sum()
        squeezed = wp.empty((self.nworld, self.hand_nq), dtype=float)
        wp.launch(squeeze_pose, (self.nworld, self.hand_nq), inputs=[self.d.qpos,
            self.d.qfrc_applied, wp.array(divisor, dtype=float), squeezed])
        self.d.qfrc_applied.zero_()
        self.trace = wp.zeros((self.nworld, int(cfg.step), self.hand_nq), dtype=float)
        self._capture("pregrasp_control", self._controls)
        self._capture("pregrasp_contacts", lambda: self._reduce_contacts(self.active))
        self._capture("pregrasp_snapshot", lambda: (
            self._copy(self.saved, to_saved=True, mask=self.newly_done),
            k.copy_world_array(self.newly_done, self.hh, self.saved_hh),
            k.copy_world_array(self.newly_done, self.ho, self.saved_ho)))
        started = time.perf_counter()
        for iteration in range(int(cfg.step)):
            self._enqueue("pregrasp_margin", lambda: wp.launch(k.prepare_pregrasp_margin,
                self.nworld, inputs=[self.active, iteration, int(cfg.step), float(cfg.cdist), self.margins, self.query]))
            self._set_margin_array()
            self._summary()
            self._launch("pregrasp_control")
            self._launch("physics", "pregrasp_physics")
            self._enqueue("pregrasp_trace", lambda: wp.launch(k.store_pregrasp_trace,
                (self.nworld, self.hand_nq), inputs=[self.active, self.d.qpos, iteration, self.trace]))
            self._launch("pregrasp_contacts")
            self._enqueue("pregrasp_stop_check", lambda: wp.launch(k.mark_grasp_done, self.nworld, inputs=[self.active, self.query, self.hh,
                self.ho, float(cfg.filter.collision.hh_thre), float(cfg.filter.collision.ho_thre),
                iteration, int(cfg.step), self.newly_done, self.iterations]))
            self._launch("pregrasp_snapshot")
        self._check()
        self.timings["pregrasp_loop_wall_s"] = time.perf_counter() - started
        self._copy(self.saved, to_saved=False)
        started = time.perf_counter()
        result = {"trace": self.trace.numpy(), "iterations": self.iterations.numpy(),
                  "qpos": self.d.qpos.numpy(), "hh": self.saved_hh.numpy(), "ho": self.saved_ho.numpy(),
                  "squeeze": squeezed.numpy()}
        self.timings["pregrasp_download_s"] = time.perf_counter() - started
        self.finish_timings()
        return result

    def finish_timings(self):
        for label, begin, end in self.events:
            self.timings["gpu_" + label + "_s"] += wp.get_event_elapsed_time(begin, end) / 1000.0
        self.events.clear()
