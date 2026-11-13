"""Optional Warp physics adapter for the unchanged grasp-refinement algorithm.

The optimizer, force mapping, filters, QP, and stopping rules remain the native
ones.  Only forward dynamics, collision detection and integration run in Warp.
The host MjData is an exported view of Warp's state, including the *pre-integration*
derived quantities left by the final step; it is never refreshed by mj_forward.
"""

from __future__ import annotations

import json
import logging

import mujoco
import numpy as np



class MuJoCoWarpAllocationError(RuntimeError):
    """A checked buffer overflow, retaining the observations needed to resize."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics

    def __reduce__(self):
        # Exceptions cross multiprocessing Pool pipes in the streaming runner.
        return type(self), (str(self), self.diagnostics)


def mixed_world_buffer_sizes(nv: int, allocation_scale: int = 1, *, refinement: bool = False) -> dict[str, int]:
    """Initial mixed-batch capacities, independent of pooled mesh geometry.

    Contact and convex-pair storage are shared across worlds. Mesh pooling can
    have many geoms and large polygons without having that many simultaneous
    collisions. Allocating the CCD polygon workspace for every potential
    contact wastes gigabytes at 640 worlds. Checked overflow causes a complete
    replay with doubled storage; no contact, physics option or world is dropped.
    """
    if int(nv) != nv or nv < 1:
        raise ValueError("nv must be a positive integer")
    if int(allocation_scale) != allocation_scale or allocation_scale < 1:
        raise ValueError("allocation_scale must be a positive integer")
    scale = int(allocation_scale)
    # Refinement includes unfiltered initial poses and 45 mm margin queries;
    # these need more convex pairs than already refined stability candidates.
    contacts, ccd_pairs, constraints = (1024, 512, 4096) if refinement else (512, 128, 2048)
    return {"nconmax": contacts * scale, "nccdmax": ccd_pairs * scale,
            "njmax": max(constraints, int(nv) * 32) * scale}


def _sync_legacy_contact_geom_fields(model, data) -> None:
    """Populate native helpers' deprecated, separately stored geom ID fields.

    MuJoCo Warp 3.17 exports only contact.geom.  In MuJoCo's C structure geom1
    and geom2 are independent fields, not aliases, so reallocating contacts can
    leave them uninitialized even though contact.geom is valid.
    """
    geom = np.asarray(data.contact.geom[: data.ncon])
    if geom.shape != (data.ncon, 2) or np.any(geom < 0) or np.any(geom >= model.ngeom):
        raise RuntimeError("MuJoCo Warp refinement exported invalid contact geometry IDs")
    data.contact.geom1[: data.ncon] = geom[:, 0]
    data.contact.geom2[: data.ncon] = geom[:, 1]


def check_warp_allocation(data, overflow_type, *, context="simulation") -> dict:
    """Reject truncated buffers, while retaining native solver-budget semantics.

    Warp reports reaching the configured solver / line-search iteration budget
    in the same bitmask as buffer overflows.  Native MuJoCo returns that finite
    iterate normally; treating these two bits as fatal would change filtering.
    No iteration count, tolerance or collision parameter is changed here.
    """
    flags = np.asarray(data.overflow.numpy(), dtype=np.int64)
    solver_mask = int(overflow_type.ITERATIONS | overflow_type.LS_ITERATIONS)
    fatal_flags = flags & ~solver_mask
    nacon = int(data.nacon.numpy()[0])
    nefc = int(np.max(data.nefc.numpy()))
    ncollision_array = getattr(data, "ncollision", None)
    ncollision = int(ncollision_array.numpy()[0]) if ncollision_array is not None else 0
    diagnostics = {
        "overflow_flags": flags.tolist(),
        "overflow_names": [
            [flag.name for flag in overflow_type if int(flag) and int(value) & int(flag)]
            for value in flags
        ],
        "allocation_overflow_flags": fatal_flags.tolist(),
        "solver_limit_flags": (flags & solver_mask).tolist(),
        "solver_limit_worlds": int(np.count_nonzero(flags & solver_mask)),
        "nacon": nacon,
        "nacon_capacity": int(data.naconmax),
        "nefc": nefc,
        "nefc_capacity": int(data.njmax),
        "ncollision": ncollision,
        "ccd_capacity": getattr(data, "naccdmax", None),
        "constraint_nnz_capacity": getattr(data, "njmax_nnz", None),
    }
    if np.any(fatal_flags) or nacon > data.naconmax or nefc > data.njmax or ncollision > data.naconmax:
        raise MuJoCoWarpAllocationError(
            f"MuJoCo Warp {context} allocation overflow: "
            + json.dumps(diagnostics, sort_keys=True),
            diagnostics,
        )
    return diagnostics


def preserve_model_options(model, warp_model) -> None:
    """Undo MJWarp's implicit tolerance change and reject lost solver options."""
    # put_model raises the native tolerance to 1e-6 for float32.  That changes an
    # experimental parameter, so explicitly retain the requested native value.
    warp_model.opt.tolerance.fill_(float(model.opt.tolerance))
    for name in (
        "integrator", "cone", "solver", "iterations", "ls_iterations",
        "noslip_iterations", "ccd_iterations", "disableflags", "enableflags",
    ):
        if not hasattr(warp_model.opt, name) or int(getattr(warp_model.opt, name)) != int(
            getattr(model.opt, name)
        ):
            raise RuntimeError(f"MuJoCo Warp did not preserve solver option {name}")
    for name in (
        "timestep", "tolerance", "ls_tolerance", "noslip_tolerance", "ccd_tolerance",
    ):
        value = getattr(warp_model.opt, name, None)
        if value is None or not np.allclose(
            value.numpy(), float(getattr(model.opt, name)), rtol=2e-7, atol=0
        ):
            raise RuntimeError(f"MuJoCo Warp did not preserve solver option {name}")


def release_warp_scene_memory(device="cuda:0") -> None:
    """Release unused graph allocations after a scene's objects are destroyed.

    Warp 1.16 synchronizes deferred graph destruction without trimming CUDA's
    graph memory pool.  Persistent workers otherwise retain each scene's high
    water mark.  This does not free live allocations or change simulation state.
    """
    import warp as wp

    device = wp.get_device(device)
    wp.synchronize_device(device)
    device.runtime.core.wp_cuda_device_graph_mem_trim(device.ordinal)
    wp.synchronize_device(device)
