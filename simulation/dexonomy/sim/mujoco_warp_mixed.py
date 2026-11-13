"""Exact per-world rigid-object models for MuJoCo Warp 3.17.

Only mesh assets are pooled. Each world has the original hand and one original
object, with its own mass, inertia, transforms and collision parameters. Missing
mesh slots use Warp's native ``geom_dataid=-1`` broadphase exclusion.
"""
from __future__ import annotations
from dexonomy.acceptance_audit import event as audit_event
from dexonomy.preparation_policy import validation_enabled

import copy
import os
import dataclasses
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class MixedModel:
    carrier_model: object
    warp_model: object
    hosts: list
    world_host_indices: np.ndarray
    geom_maps: list
    inverse_geom_maps: list
    metadata: dict

    def initialize_data(self, data):
        """Seed static transforms, which Warp intentionally skips in kinematics."""
        import warp as wp
        for name in ("xpos", "xquat", "xmat", "xipos", "ximat", "subtree_com"):
            dest = getattr(data, name)
            shape = dest.numpy().shape[1:]
            values = [np.asarray(getattr(h._data, name)).reshape(shape) for h in self.hosts]
            dest.assign(np.stack(values)[self.world_host_indices])
        for name in ("geom_xpos", "geom_xmat"):
            dest = getattr(data, name)
            shape = dest.numpy().shape[1:]
            values = []
            for h, mapping in zip(self.hosts, self.geom_maps):
                value = np.zeros(shape, np.float32)
                value[mapping] = np.asarray(getattr(h._data, name)).reshape((h._model.ngeom,) + shape[1:])
                values.append(value)
            dest.assign(np.stack(values)[self.world_host_indices])

    def original_geoms(self, packed, world):
        """Translate packed contact IDs back to the candidate's native model."""
        selected = np.flatnonzero(packed["worldid"] == world)
        result = {name: np.array(value[selected], copy=True) for name, value in packed.items()}
        inverse = self.inverse_geom_maps[int(self.world_host_indices[world])]
        result["geom"] = inverse[result["geom"]]
        if np.any(result["geom"] < 0):
            raise RuntimeError("MuJoCo Warp emitted contact for an absent mesh slot")
        result["worldid"].fill(0)
        return result


def _geom_key(model, index):
    return tuple(int(getattr(model, name)[index]) for name in (
        "geom_type", "geom_contype", "geom_conaffinity", "geom_condim", "geom_group", "geom_priority"))


def _folded_damping(model, name):
    """Match put_model's actuator damping fold before per-world upload."""
    value = np.array(getattr(model, name), copy=True)
    poly = name.endswith("poly")
    is_tendon = name.startswith("tendon")
    damping = model.actuator_dampingpoly if poly else model.actuator_damping
    for actuator in range(model.nu):
        trn = int(model.actuator_trntype[actuator])
        joint = int(model.actuator_trnid[actuator, 0])
        addition = damping[actuator] * float(model.actuator_gear[actuator, 0]) ** 2
        if is_tendon and trn == int(mujoco.mjtTrn.mjTRN_TENDON):
            value[joint] += addition
        elif not is_tendon and trn in (int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)):
            value[model.dof_jntid == joint] += addition
    return value


def _prepare_carrier(hosts, *, timings=None):
    """CPU-only topology normalization; preserve every source model untouched."""
    started = time.perf_counter()
    timings = {} if timings is None else timings
    if not hosts:
        raise ValueError("mixed model requires at least one scene")
    base = hosts[0]._model
    object_root = int(hosts[0]._obj_active_bodyid)
    object_bodies = np.unique(base.geom_bodyid[base.geom_bodyid >= object_root])
    if len(object_bodies) != 1:
        raise ValueError("mixed scenes require object meshes on one rigid body")
    object_body = int(object_bodies[0])
    common = np.flatnonzero(base.geom_bodyid != object_body)
    if np.any(common != np.arange(len(common))):
        raise ValueError("mixed rigid-object scenes require object geoms after hand geoms")
    structural = (
        "body_parentid", "body_rootid", "body_weldid", "body_mocapid", "body_jntnum", "body_jntadr",
        "body_dofnum", "body_dofadr", "body_treeid", "jnt_type", "jnt_qposadr", "jnt_dofadr",
        "jnt_bodyid", "jnt_limited", "dof_bodyid", "dof_jntid", "dof_parentid", "dof_treeid", "dof_Madr",
        "actuator_trntype", "actuator_trnid", "actuator_dyntype", "actuator_gaintype", "actuator_biastype",
        "eq_type", "eq_obj1id", "eq_obj2id", "exclude_signature",
    )
    counts, groups, maximum = [], [], Counter()
    for host in hosts:
        model = host._model
        for size in ("nq", "nv", "nu", "na", "nbody", "njnt", "neq", "nmocap", "ntendon", "nsite", "nsensor"):
            if getattr(model, size) != getattr(base, size):
                raise ValueError(f"mixed scenes differ in topology: {size}")
        if host._obj_active_bodyid != object_root or object_body != base.nbody - 1:
            raise ValueError("mixed scenes require exactly one terminal rigid-object body")
        for name in structural:
            if not np.array_equal(getattr(base, name), getattr(model, name)):
                raise ValueError(f"mixed scenes differ in non-batched topology: {name}")
        if model.npair or model.nflex or model.nhfield:
            raise ValueError("mixed scenes do not support explicit geom pairs, flexes or heightfields")
        if int(model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_SLEEP):
            raise ValueError("mixed scenes do not support sleep")
        before = np.flatnonzero(model.geom_bodyid != object_body)
        if not np.array_equal(before, common):
            raise ValueError("mixed scenes must share the complete hand geometry")
        for name in ("geom_type", "geom_contype", "geom_conaffinity", "geom_condim", "geom_bodyid", "geom_group", "geom_priority"):
            if not np.array_equal(getattr(base, name)[common], getattr(model, name)[common]):
                raise ValueError(f"mixed hand geometry differs: {name}")
        by_group = {}
        for geom in np.flatnonzero(model.geom_bodyid == object_body):
            if model.geom_type[geom] != int(mujoco.mjtGeom.mjGEOM_MESH):
                raise ValueError("mixed object geometry must consist of meshes")
            by_group.setdefault(_geom_key(model, geom), []).append(int(geom))
        count = Counter({key: len(indices) for key, indices in by_group.items()})
        maximum |= count
        counts.append(count)
        groups.append(by_group)

    # body_simple is a compiler optimization flag, not rigid-body topology.
    # Some DGN objects have centered principal frames (1), others do not (0).
    # Both use Warp's general rigid-body path; a shared 0 keeps that path valid
    # without changing any native model or per-world inertial parameters.
    body_simple = np.stack([host._model.body_simple for host in hosts])
    differs = np.any(body_simple != body_simple[0], axis=0)
    if np.any(differs & np.any((body_simple != 0) & (body_simple != 1), axis=0)):
        raise ValueError("mixed scenes differ in unsupported body_simple optimization")
    shared_body_simple = np.where(differs, 0, body_simple[0])

    timings["topology_s"] = time.perf_counter() - started
    phase = time.perf_counter()
    tree = ET.fromstring(hosts[0]._spec.to_xml())
    timings["xml_roundtrip_s"] = time.perf_counter() - phase
    phase = time.perf_counter()
    asset = tree.find("asset")
    if asset is None:
        asset = ET.SubElement(tree, "asset")
    mesh_names = []
    extra_meshes = []
    for index, host in enumerate(hosts):
        mapping = {}
        for mesh in host._spec.meshes:
            name = mesh.name
            if index == 0 or name.startswith(host._cfg.hand_prefix):
                mapping[name] = name
            else:
                mapping[name] = f"v13_mesh_{index}_{name}"
                extra_meshes.append((mapping[name], mesh))
        mesh_names.append(mapping)
    object_name = base.body(object_body).name
    body = next(node for node in tree.iter("body") if node.get("name") == object_name)
    for geom in list(body.findall("geom")):
        body.remove(geom)
    slots = {}
    slot = len(common)
    for key in sorted(maximum):
        slots[key] = list(range(slot, slot + maximum[key]))
        donor = next(i for i, group in enumerate(groups) if key in group)
        model = hosts[donor]._model
        mesh = model.mesh(int(model.geom_dataid[groups[donor][key][0]])).name
        for index in slots[key]:
            ET.SubElement(body, "geom", name=f"v13_object_slot_{index}", type="mesh",
                          mesh=mesh_names[donor][mesh], contype=str(key[1]), conaffinity=str(key[2]),
                          condim=str(key[3]), group=str(key[4]), priority=str(key[5]))
        slot += maximum[key]
    timings["asset_merge_including_xml_s"] = time.perf_counter() - phase
    phase = time.perf_counter()
    carrier_spec = mujoco.MjSpec.from_string(ET.tostring(tree, encoding="unicode"))
    timings["xml_roundtrip_s"] += time.perf_counter() - phase
    phase = time.perf_counter()
    # Copy resolved mesh declarations directly. Keep the first scene's XML
    # roundtrip unchanged so carrier topology/float serialization stays fixed.
    fields = ("content_type", "file", "refpos", "refquat", "scale", "inertia",
              "smoothnormal", "needsdf", "maxhullvert", "uservert", "usernormal",
              "usertexcoord", "userface", "userfacenormal", "userfacetexcoord",
              "usernode", "usertet", "material", "octree_maxdepth")
    for name, mesh in extra_meshes:
        if mesh.needsdf:
            raise ValueError("direct mesh pooling does not support SDF plugins")
        carrier_spec.add_mesh(name=name, **{field: getattr(mesh, field) for field in fields})
    timings["direct_mesh_merge_s"] = time.perf_counter() - phase
    phase = time.perf_counter()
    pooled_meshes = {mesh.name: mesh for mesh in carrier_spec.meshes}
    used_collision_meshes = {geom.meshname for geom in carrier_spec.geoms
                             if geom.contype or geom.conaffinity}
    # to_xml serializes floats at limited precision. Restore the exact source
    # mesh parameters before compilation: even sub-micrometre scale rounding
    # can change GJK/EPA branches and the MULTICCD contact manifold.
    mesh_parameters = ("scale", "refpos", "refquat", "uservert", "usernormal", "usertexcoord",
                       "userface", "userfacenormal", "userfacetexcoord", "usernode", "usertet")
    restored_parameters = {}
    for hindex, host in enumerate(hosts):
        for original in host._spec.meshes:
            pooled = pooled_meshes[mesh_names[hindex][original.name]]
            signature = tuple(np.asarray(getattr(original, name)).tobytes() for name in mesh_parameters)
            previous = restored_parameters.get(pooled.name)
            if previous is not None:
                if signature != previous:
                    raise ValueError("shared mesh name has different source parameters")
                continue
            restored_parameters[pooled.name] = signature
            for name in mesh_parameters:
                setattr(pooled, name, getattr(original, name))
            native_id = host._model.mesh(original.name).id
            if host._model.mesh_graphadr[native_id] >= 0 and pooled.name not in used_collision_meshes:
                # Meshes used only by another world's dataid are otherwise
                # "unused" to the compiler, which omits their GJK/MULTICCD
                # hull data. Convex inertia explicitly requests that hull.
                pooled.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX
    timings["mesh_parameter_restore_s"] = time.perf_counter() - phase
    phase = time.perf_counter()
    carrier = carrier_spec.compile()
    timings["carrier_compile_s"] = time.perf_counter() - phase
    phase = time.perf_counter()
    copied = {}
    copy_count = skipped = 0
    verify_s = 0.0
    from .mujoco_warp_mesh_pool import copy_native_mesh, native_mesh_identity
    for hindex, host in enumerate(hosts):
        original = host._model
        for mid in range(original.nmesh):
            pooled_id = carrier.mesh(mesh_names[hindex][original.mesh(mid).name]).id
            # A pooled name alone is not evidence of equal native collision data.
            identity = native_mesh_identity(original, mid)
            if pooled_id in copied:
                if copied[pooled_id] != identity:
                    raise ValueError("shared mesh name has different native collision data")
                skipped += 1
            else:
                copy_native_mesh(original, mid, carrier, pooled_id)
                copied[pooled_id] = identity
                copy_count += 1
            verify_start = time.perf_counter()
            old_start, old_count = original.mesh_vertadr[mid], original.mesh_vertnum[mid]
            new_start, new_count = carrier.mesh_vertadr[pooled_id], carrier.mesh_vertnum[pooled_id]
            if old_count != new_count or (validation_enabled() and not np.array_equal(
                    original.mesh_vert[old_start:old_start + old_count],
                    carrier.mesh_vert[new_start:new_start + new_count])):
                raise RuntimeError(f"mixed mesh pool changed compiled source mesh vertices: {original.mesh(mid).name}")
            if original.mesh_graphadr[mid] >= 0 and (
                    carrier.mesh_graphadr[pooled_id] < 0 or
                    carrier.mesh_polynum[pooled_id] != original.mesh_polynum[mid]):
                raise RuntimeError("mixed mesh pool omitted source convex collision hull")
            verify_s += time.perf_counter() - verify_start
    timings["mesh_copy_identity_and_verify_s"] = time.perf_counter() - phase
    timings["mesh_verify_nested_s"] = verify_s
    timings["mesh_copies"] = copy_count
    timings["mesh_duplicate_copies_skipped"] = skipped
    phase = time.perf_counter()
    for size in ("npolygonmax", "nmeshdegmax"):
        if int(getattr(carrier, size)) < max(int(getattr(h._model, size)) for h in hosts):
            raise RuntimeError(f"mixed mesh pool has insufficient CCD scratch size: {size}")
    for name in structural:
        getattr(carrier, name)[:] = getattr(base, name)
    carrier.body_simple[:] = shared_body_simple

    maps, inverse = [], []
    for host, group in zip(hosts, groups):
        mapping = np.empty(host._model.ngeom, dtype=np.int32)
        mapping[common] = common
        for key, geoms in group.items():
            mapping[geoms] = slots[key][:len(geoms)]
        back = np.full(carrier.ngeom, -1, dtype=np.int32)
        back[mapping] = np.arange(host._model.ngeom)
        maps.append(mapping)
        inverse.append(back)
    timings["mapping_s"] = time.perf_counter() - phase
    timings["carrier_total_s"] = time.perf_counter() - started
    audit_event("model_preparation", validation=validation_enabled(), timings=timings)
    return carrier, maps, inverse, mesh_names


def build_mixed_model(hosts, world_host_indices, *, device="cuda:0"):
    """Upload equivalent native models to independent worlds of one Warp batch."""
    import mujoco_warp as mjw
    from mujoco_warp._src import types
    import warp as wp
    from .mujoco_warp_env import preserve_model_options

    wp.set_device(device)
    indices = np.asarray(world_host_indices, dtype=np.int32)
    if indices.ndim != 1 or not len(indices) or np.any(indices < 0) or np.any(indices >= len(hosts)):
        raise ValueError("invalid world-to-scene mapping")
    carrier, maps, inverse, mesh_names = _prepare_carrier(hosts)
    return _upload_mixed_model(hosts, indices, carrier, maps, inverse, mesh_names, device=device)


def _upload_mixed_model(hosts, indices, carrier, maps, inverse, mesh_names, *, device):
    """Upload a prepared carrier; shared by the original path and the bank."""
    import mujoco_warp as mjw
    from mujoco_warp._src import types
    import warp as wp
    from .mujoco_warp_env import preserve_model_options

    model = mjw.put_model(carrier)
    option_names = ("integrator", "cone", "solver", "iterations", "ls_iterations", "noslip_iterations",
                    "ccd_iterations", "disableflags", "enableflags", "timestep", "tolerance",
                    "ls_tolerance", "noslip_tolerance", "ccd_tolerance", "impratio")
    for name in option_names:
        values = [getattr(host._model.opt, name) for host in hosts]
        if any(value != values[0] for value in values):
            raise ValueError(f"mixed model option differs between worlds: {name}")
    # Keep native model options, including the original 1e-8 solver tolerance.
    preserve_model_options(hosts[0]._model, model)
    uploaded = []
    for field in dataclasses.fields(types.Model):
        spec = getattr(field.type, "shape", ())
        name = field.name
        if not spec or spec[0] != "*" or not hasattr(hosts[0]._model, name):
            continue
        dest = getattr(model, name)
        if not dest.size:
            continue
        shape = dest.numpy().shape[1:]
        values = []
        for hindex, (host, mapping) in enumerate(zip(hosts, maps)):
            original = np.asarray(getattr(host._model, name))
            if name in ("dof_damping", "dof_dampingpoly", "tendon_damping", "tendon_dampingpoly"):
                original = _folded_damping(host._model, name)
            if name.startswith("geom_"):
                value = np.zeros(shape, dtype=original.dtype)
                value[mapping] = original.reshape((host._model.ngeom,) + shape[1:])
                if name == "geom_dataid":
                    value.fill(-1)
                    for old, new in enumerate(mapping):
                        mesh_id = int(host._model.geom_dataid[old])
                        if mesh_id >= 0:
                            mesh_name = host._model.mesh(mesh_id).name
                            value[new] = carrier.mesh(mesh_names[hindex][mesh_name]).id
                elif name == "geom_quat":
                    value[inverse[hindex] < 0, 0] = 1
            else:
                value = original.reshape(shape)
            values.append(value)
        different = name == "geom_dataid" or any(not np.array_equal(values[0], v) for v in values[1:])
        array = np.stack(values)[indices] if different else values[0][None]
        setattr(model, name, wp.array(array, dtype=dest.dtype, device=device))
        if different:
            uploaded.append(name)
    metadata = {
        "active_worlds": len(indices), "unique_scenes": len(hosts), "carrier_geoms": carrier.ngeom,
        "scene_geoms": [h._model.ngeom for h in hosts], "mesh_assets": carrier.nmesh,
        "per_world_fields": uploaded, "physics_profile": "native", "strict_tolerance": True,
        "model_options": {name: float(getattr(hosts[0]._model.opt, name)) for name in option_names},
        "missing_mesh_slots": int(sum(np.count_nonzero(inverse[int(i)] < 0) for i in indices)),
    }
    return MixedModel(carrier, model, list(hosts), indices, maps, inverse, metadata)


@dataclass
class _BankBoundMixedModel(MixedModel):
    """One batch's mutable parameters, backed by immutable pooled GPU assets."""

    bank: object
    device_world_host_indices: object

    def initialize_data(self, data):
        # Host MjData is subsequently reused by the CPU QP/filter. Its derived
        # transforms are mutable, so always gather the bank's startup snapshot.
        import warp as wp
        for name, table in self.bank._initial_transforms.items():
            target = getattr(data, name)
            if target.shape != (len(self.world_host_indices),) + table.shape[1:]:
                raise ValueError(f"model bank transform shape differs: {name}")
            wp.copy(target, wp.indexedarray(table, [self.device_world_host_indices]))


def _gpu_array_bytes(value):
    """Count unique resident arrays, including nested model options."""
    import warp as wp
    from warp._src.types import type_size_in_bytes

    seen = set()

    def visit(item):
        if isinstance(item, wp.array):
            key = (str(item.device), item.ptr)
            if key in seen:
                return 0
            seen.add(key)
            return int(item.size) * int(type_size_in_bytes(item.dtype))
        if dataclasses.is_dataclass(item):
            return sum(visit(getattr(item, field.name)) for field in dataclasses.fields(item))
        if isinstance(item, dict):
            return sum(visit(child) for child in item.values())
        return 0

    return visit(value)


class MixedModelBank:
    """Prepare all scene meshes and model parameters once, then bind on GPU.

    The host inventory must contain freshly initialized environments with the
    same topology and options. Source models/data remain untouched. The bank
    snapshots their initial parameters/transforms at construction; later host
    QP/filter mutations do not affect subsequent batches. Only world-to-scene
    indices cross from CPU to GPU in :meth:`bind`; static mesh assets remain
    shared on the device, while all mutable batch fields own their storage.
    """

    def __init__(self, hosts, *, device="cuda:0"):
        import warp as wp

        started = time.perf_counter()
        self.hosts = list(hosts)
        self.device = str(device)
        self.bind_count = 0
        wp.set_device(device)
        preparation_timings = {}
        carrier, maps, inverse, mesh_names = _prepare_carrier(self.hosts, timings=preparation_timings)
        prepared = time.perf_counter()
        template = _upload_mixed_model(
            self.hosts, np.arange(len(self.hosts), dtype=np.int32),
            carrier, maps, inverse, mesh_names, device=device)
        wp.synchronize_device(device)
        uploaded = time.perf_counter()
        self.carrier_model = carrier
        self._template = template
        self._parameter_fields = tuple(template.metadata["per_world_fields"])
        self._native_expected = None
        if os.environ.get("DEXONOMY_VALIDATE_WORLD_ROWS") == "1":
            from dexonomy.sim.world_rows import native_rows
            self._native_expected = native_rows(self, mesh_names)
        self._initial_transforms = {}
        body_fields = {"xpos": wp.vec3, "xquat": wp.quat, "xmat": wp.mat33,
                       "xipos": wp.vec3, "ximat": wp.mat33, "subtree_com": wp.vec3}
        for name, dtype in body_fields.items():
            trailing = (3, 3) if dtype == wp.mat33 else ((4,) if dtype == wp.quat else (3,))
            values = [np.asarray(getattr(host._data, name)).reshape((carrier.nbody,) + trailing)
                      for host in self.hosts]
            self._initial_transforms[name] = wp.array(np.stack(values), dtype=dtype, device=device)
        for name, dtype, trailing in (("geom_xpos", wp.vec3, (3,)),
                                      ("geom_xmat", wp.mat33, (3, 3))):
            values = []
            for host, mapping in zip(self.hosts, maps):
                value = np.zeros((carrier.ngeom,) + trailing, dtype=np.float32)
                value[mapping] = np.asarray(getattr(host._data, name)).reshape(
                    (host._model.ngeom,) + trailing)
                values.append(value)
            self._initial_transforms[name] = wp.array(np.stack(values), dtype=dtype, device=device)
        wp.synchronize_device(device)
        finished = time.perf_counter()
        parameter_bytes = _gpu_array_bytes({name: getattr(template.warp_model, name)
                                            for name in self._parameter_fields})
        model_bytes = _gpu_array_bytes(template.warp_model)
        transform_bytes = _gpu_array_bytes(self._initial_transforms)
        self.metadata = {
            "preparation_breakdown": preparation_timings,
            "scene_count": len(self.hosts), "bank_scenes": len(self.hosts),
            "carrier_model_builds": 1, "static_upload_count": 1,
            "duration_s": finished - started,
            "carrier_prepare_s": prepared - started,
            "parameter_pack_upload_s": uploaded - prepared,
            "initial_transform_upload_s": finished - uploaded,
            "mixed_model_pack_upload_s": finished - started,
            "model_gpu_bytes": model_bytes,
            "per_scene_parameter_gpu_bytes": parameter_bytes,
            "static_model_gpu_bytes": model_bytes - parameter_bytes,
            "initial_transform_gpu_bytes": transform_bytes,
            "resident_gpu_bytes": model_bytes + transform_bytes,
            "carrier_geoms": carrier.ngeom, "mesh_assets": carrier.nmesh,
            "per_scene_parameter_fields": list(self._parameter_fields),
        }

    @property
    def cache_stats(self):
        return {**self.metadata, "bind_count": self.bind_count}

    def bind(self, world_host_indices):
        """Select scene rows on-device without rebuilding/uploading meshes."""
        import warp as wp

        started = time.perf_counter()
        indices = np.array(world_host_indices, dtype=np.int32, copy=True)
        if (indices.ndim != 1 or not len(indices) or np.any(indices < 0)
                or np.any(indices >= len(self.hosts))):
            raise ValueError("invalid world-to-scene mapping")
        wp.set_device(self.device)
        device_indices = wp.array(indices, dtype=wp.int32, device=self.device)
        template = self._template
        model = copy.copy(template.warp_model)
        # preserve_model_options fills opt.tolerance. Give every batch its own
        # option arrays as well, so even an accidental batch mutation cannot
        # change the startup snapshot used by a later bind.
        model.opt = copy.copy(template.warp_model.opt)
        for field in dataclasses.fields(model.opt):
            value = getattr(model.opt, field.name)
            if isinstance(value, wp.array):
                setattr(model.opt, field.name, wp.clone(value))
        fields = set(self._parameter_fields) | {"geom_margin"}
        bound_bytes = 0
        for name in sorted(fields):
            table = getattr(template.warp_model, name)
            target = wp.empty((len(indices),) + table.shape[1:], dtype=table.dtype, device=self.device)
            # A common geom_margin is stored in one row; create a stride-zero
            # broadcast view on the device instead of uploading repeated rows.
            if table.shape[0] == 1:
                source = wp.array(ptr=table.ptr, dtype=table.dtype, device=self.device,
                                  shape=target.shape, strides=(0,) + table.strides[1:])
            else:
                source = wp.indexedarray(table, [device_indices])
            wp.copy(target, source)
            setattr(model, name, target)
            bound_bytes += _gpu_array_bytes(target)
        self.bind_count += 1
        metadata = {
            **template.metadata,
            "active_worlds": len(indices), "unique_scenes": int(len(np.unique(indices))),
            "bank_scenes": len(self.hosts), "model_cache_mode": "preloaded",
            "missing_mesh_slots": int(sum(np.count_nonzero(template.inverse_geom_maps[int(i)] < 0)
                                          for i in indices)),
            "model_bank_bind_count": self.bind_count,
            "model_bank_bind_submit_s": time.perf_counter() - started,
            "model_bank_index_upload_bytes": int(indices.nbytes),
            "model_bank_parameter_gather_bytes": bound_bytes,
            "model_bank_static_upload_bytes": 0,
        }
        # One real world can still contain absent slots from a larger bank.
        # MJWarp 3.17 gates missing-mesh filtering on dataid row count > 1.
        # Duplicate only this lookup row, not worlds, candidates, or MjData.
        if len(indices) == 1 and metadata["missing_mesh_slots"]:
            row = model.geom_dataid
            table = wp.empty((2,) + row.shape[1:], dtype=row.dtype, device=self.device)
            wp.copy(table[0:1], row)
            wp.copy(table[1:2], row)
            model.geom_dataid = table
            metadata["singleton_dataid_compat_rows"] = 2
            metadata["model_bank_parameter_gather_bytes"] += _gpu_array_bytes(row)
        if self._native_expected is not None:
            from dexonomy.sim.world_rows import verify_rows
            checked_started=time.perf_counter()
            metadata["actual_world_parameters_checked"]=verify_rows(self._native_expected, model, indices)
            metadata["world_parameter_check_s"]=time.perf_counter()-checked_started
        audit_event("model_bind",metadata=metadata,world_scene_indices=indices,geom_dataid_rows=model.geom_dataid.shape[0])
        return _BankBoundMixedModel(
            self.carrier_model, model, self.hosts, indices, template.geom_maps,
            template.inverse_geom_maps, metadata, self, device_indices)
