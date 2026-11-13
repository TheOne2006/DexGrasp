"""Device-resident helpers for the ordinary ``op=grasp`` refinement loop.

The caller owns MuJoCo Warp models/data, graph capture, and phase scheduling.
All arrays are already on the selected device; these helpers never download
state or synchronize.  A batch shares one scene and pads template contacts to
``K`` slots, with ``counts[world]`` identifying valid slots.

Spring forces are mapped to generalized forces *once per outer iteration* by
the caller with ``mujoco_warp.xfrc_accumulate``.  Clear ``xfrc_applied`` after
that mapping, before the physics substeps: keeping Cartesian forces live would
change the reference algorithm's fixed-generalized-force timing.
"""

from typing import Any

import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.kernel
def initialize_local_points(
    body_ids: wp.array2d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    world_points: wp.array2d(dtype=wp.vec3),
    xpos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33),
    local_points: wp.array2d(dtype=wp.vec3),
):
    """Launch ``(B, K)`` after the candidate's initial forward operation."""
    world, contact = wp.tid()
    if contact < counts[world]:
        body = body_ids[world, contact]
        local_points[world, contact] = wp.transpose(xmat[world, body]) @ (
            world_points[world, contact] - xpos[world, body]
        )
    else:
        local_points[world, contact] = wp.vec3(0.0)


@wp.kernel
def prepare_controls(
    active: wp.array(dtype=wp.int32),
    qpos: wp.array2d(dtype=wp.float32),
    ctrl_map: wp.array2d(dtype=wp.float32),
    ctrl: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    qfrc: wp.array2d(dtype=wp.float32),
    xfrc: wp.array2d(dtype=wp.spatial_vector),
):
    """Launch ``B``; reproduce current-pose controls and zero velocities/forces.

    ``ctrl_map`` is the host ``_hand_qpos2ctrl_mat[:, 6:]``.  This kernel is for
    the free-hand/mocap refinement scene; the caller must reject arm/IK scenes.
    It deliberately does not update the original wrist mocap target.
    """
    world = wp.tid()
    if active[world] != 0:
        for actuator in range(ctrl.shape[1]):
            target = float(0.0)
            for joint in range(ctrl_map.shape[1]):
                target += ctrl_map[actuator, joint] * qpos[world, joint + 7]
            ctrl[world, actuator] = target
        for dof in range(qvel.shape[1]):
            qvel[world, dof] = 0.0
            qfrc[world, dof] = 0.0
        for body in range(xfrc.shape[1]):
            xfrc[world, body] = wp.spatial_vector()


@wp.kernel
def spring_body_wrenches(
    active: wp.array(dtype=wp.int32),
    body_ids: wp.array2d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    local_points: wp.array2d(dtype=wp.vec3),
    targets: wp.array2d(dtype=wp.vec3),
    normals: wp.array2d(dtype=wp.vec3),
    xpos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33),
    xipos: wp.array2d(dtype=wp.vec3),
    current_dist: wp.array2d(dtype=wp.float32),
    xfrc: wp.array2d(dtype=wp.spatial_vector),
):
    """Launch ``(B, nbody)``; sum contacts in their original order per body.

    MuJoCo's applied Cartesian wrench is force then torque, both in world
    coordinates, with the torque measured about the body's inertial COM.
    Target normals are used as supplied, matching the NumPy implementation.
    """
    world, body = wp.tid()
    if active[world] != 0:
        force_sum = wp.vec3(0.0)
        torque_sum = wp.vec3(0.0)
        for contact in range(counts[world]):
            if body_ids[world, contact] == body:
                point = xmat[world, body] @ local_points[world, contact] + xpos[world, body]
                delta = targets[world, contact] - point
                normal = normals[world, contact]
                normal_delta = wp.dot(delta, normal) * normal
                force = normal_delta * 500.0 + (delta - normal_delta) * 100.0
                current_dist[world, contact] = wp.length(delta)
                force_sum += force
                torque_sum += wp.cross(point - xipos[world, body], force)
        xfrc[world, body] = wp.spatial_vector(force_sum, torque_sum)


@wp.kernel
def update_progress(
    active: wp.array(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    current_dist: wp.array2d(dtype=wp.float32),
    previous_dist: wp.array2d(dtype=wp.float32),
    iteration: int,
    needs_query: wp.array(dtype=wp.int32),
):
    """Launch ``B``; request collision forward only after progress plateaus."""
    world = wp.tid()
    needs_query[world] = 0
    if active[world] != 0:
        improvement = float(-1.0e30)
        for contact in range(counts[world]):
            improvement = wp.max(
                improvement, previous_dist[world, contact] - current_dist[world, contact]
            )
            previous_dist[world, contact] = current_dist[world, contact]
        if iteration > 0 and improvement <= 1.0e-4:
            needs_query[world] = 1


@wp.kernel
def reduce_contact_distances(
    query_mask: wp.array(dtype=wp.int32),
    nacon: wp.array(dtype=wp.int32),
    contact_worldid: wp.array(dtype=wp.int32),
    contact_geom: wp.array(dtype=wp.vec2i),
    contact_dist: wp.array(dtype=wp.float32),
    contact_type: wp.array(dtype=wp.int32),
    exported_type_mask: int,
    geom_bodyid: wp.array(dtype=wp.int32),
    hand_body_count: int,
    mocap_count: int,
    hh_min: wp.array(dtype=wp.float32),
    ho_min: wp.array(dtype=wp.float32),
):
    """Launch ``B``; match ``get_contacts`` classification on packed contacts.

    Empty contact sets have infinite minimum distance and pass the collision
    filter, as in the native code.  This reduction does not enforce required
    contacts, joint limits, or the final QP.  The caller separately checks
    allocation overflow; clamping here only prevents an out-of-bounds read.
    """
    world = wp.tid()
    hand_hand = float(1.0e30)
    hand_object = float(1.0e30)
    if query_mask[world] != 0:
        object_id = hand_body_count + mocap_count
        count = wp.min(nacon[0], contact_dist.shape[0])
        for contact in range(count):
            if contact_worldid[contact] == world and (contact_type[contact] & exported_type_mask) != 0:
                pair = contact_geom[contact]
                if pair[0] >= 0 and pair[1] >= 0:
                    body1 = geom_bodyid[pair[0]]
                    body2 = geom_bodyid[pair[1]]
                    hand1 = body1 > 0 and body1 <= hand_body_count
                    hand2 = body2 > 0 and body2 <= hand_body_count
                    if (hand1 and body2 > object_id) or (hand2 and body1 > object_id):
                        hand_object = wp.min(hand_object, contact_dist[contact])
                    elif hand1 and hand2:
                        hand_hand = wp.min(hand_hand, contact_dist[contact])
    hh_min[world] = hand_hand
    ho_min[world] = hand_object


@wp.kernel
def initialize_contact_minima(
    hh_min: wp.array(dtype=wp.float32),
    ho_min: wp.array(dtype=wp.float32),
):
    """Launch ``B`` before the contact-parallel reduction, including idle worlds."""
    world = wp.tid()
    hh_min[world] = 1.0e30
    ho_min[world] = 1.0e30


@wp.kernel
def reduce_contact_distances_parallel(
    query_mask: wp.array(dtype=wp.int32),
    nacon: wp.array(dtype=wp.int32),
    contact_worldid: wp.array(dtype=wp.int32),
    contact_geom: wp.array(dtype=wp.vec2i),
    contact_dist: wp.array(dtype=wp.float32),
    contact_type: wp.array(dtype=wp.int32),
    exported_type_mask: int,
    geom_bodyid: wp.array(dtype=wp.int32),
    hand_body_count: int,
    mocap_count: int,
    hh_min: wp.array(dtype=wp.float32),
    ho_min: wp.array(dtype=wp.float32),
):
    """Launch ``naconmax`` after :func:`initialize_contact_minima`.

    Each valid packed contact contributes once to its world's minimum. Padded
    slots, unqueried worlds, and non-exported contacts do not contribute. The
    caller still checks allocation overflow independently of this reduction.
    """
    contact = wp.tid()
    if contact >= nacon[0] or contact >= contact_dist.shape[0]:
        return
    world = contact_worldid[contact]
    if world < 0 or world >= query_mask.shape[0]:
        return
    if query_mask[world] == 0 or (contact_type[contact] & exported_type_mask) == 0:
        return
    pair = contact_geom[contact]
    if pair[0] < 0 or pair[1] < 0:
        return
    if pair[0] >= geom_bodyid.shape[0] or pair[1] >= geom_bodyid.shape[0]:
        return
    body1 = geom_bodyid[pair[0]]
    body2 = geom_bodyid[pair[1]]
    hand1 = body1 > 0 and body1 <= hand_body_count
    hand2 = body2 > 0 and body2 <= hand_body_count
    object_id = hand_body_count + mocap_count
    if (hand1 and body2 > object_id) or (hand2 and body1 > object_id):
        wp.atomic_min(ho_min, world, contact_dist[contact])
    elif hand1 and hand2:
        wp.atomic_min(hh_min, world, contact_dist[contact])


@wp.kernel
def mark_grasp_done(
    active: wp.array(dtype=wp.int32),
    needs_query: wp.array(dtype=wp.int32),
    hh_min: wp.array(dtype=wp.float32),
    ho_min: wp.array(dtype=wp.float32),
    hh_threshold: float,
    ho_threshold: float,
    iteration: int,
    max_iterations: int,
    newly_done: wp.array(dtype=wp.int32),
    iterations: wp.array(dtype=wp.int32),
):
    """Launch ``B``; signal the first stopping state for snapshot kernels.

    This also handles pregrasp when ``needs_query`` is set by
    :func:`prepare_pregrasp_margin`.  Hitting the iteration cap records a final
    state; it does not imply that the candidate passed the final filters.
    """
    world = wp.tid()
    newly_done[world] = 0
    if active[world] != 0:
        collision_free = hh_min[world] >= hh_threshold and ho_min[world] >= ho_threshold
        if (needs_query[world] != 0 and collision_free) or iteration + 1 >= max_iterations:
            newly_done[world] = 1
            iterations[world] = iteration + 1
            active[world] = 0


@wp.kernel
def fill_world_margins(
    value: float,
    margins: wp.array(dtype=wp.float32),
):
    world = wp.tid()
    margins[world] = value


@wp.kernel
def set_object_margin(
    object_geom_mask: wp.array(dtype=wp.int32),
    margins: wp.array(dtype=wp.float32),
    geom_margin: wp.array2d(dtype=wp.float32),
):
    """Launch ``(B, ngeom)`` on a model with per-world geometry margins."""
    world, geom = wp.tid()
    if object_geom_mask[geom] != 0:
        geom_margin[world, geom] = margins[world]


@wp.kernel
def prepare_pregrasp_margin(
    active: wp.array(dtype=wp.int32),
    iteration: int,
    max_iterations: int,
    target_margin: float,
    margins: wp.array(dtype=wp.float32),
    needs_query: wp.array(dtype=wp.int32),
):
    """Launch ``B`` before control reset and pregrasp physics substeps."""
    world = wp.tid()
    needs_query[world] = 0
    if active[world] != 0:
        fraction = wp.min(float(iteration + 1) / float(max_iterations) * 2.0, 1.0)
        margin = target_margin * fraction
        margins[world] = margin
        if iteration % 2 == 0 and margin >= target_margin:
            needs_query[world] = 1


@wp.kernel
def store_pregrasp_trace(
    active: wp.array(dtype=wp.int32),
    qpos: wp.array2d(dtype=wp.float32),
    iteration: int,
    trace: wp.array3d(dtype=wp.float32),
):
    """Launch ``(B, hand_nq)`` after stepping and before marking worlds done.

    Trace shape is ``(B, max_iterations, hand_nq)``; the caller uses each
    world's recorded iteration count before applying native ``[::-2]`` order.
    """
    world, dof = wp.tid()
    if active[world] != 0:
        trace[world, iteration, dof] = qpos[world, dof]


@wp.kernel
def masked_copy_1d(
    mask: wp.array(dtype=wp.int32),
    source: wp.array(dtype=Any),
    destination: wp.array(dtype=Any),
):
    world = wp.tid()
    if mask[world] != 0:
        destination[world] = source[world]


@wp.kernel
def masked_copy_2d(
    mask: wp.array(dtype=wp.int32),
    source: wp.array2d(dtype=Any),
    destination: wp.array2d(dtype=Any),
):
    world, element = wp.tid()
    if mask[world] != 0:
        destination[world, element] = source[world, element]


@wp.kernel
def masked_copy_3d(
    mask: wp.array(dtype=wp.int32),
    source: wp.array3d(dtype=Any),
    destination: wp.array3d(dtype=Any),
):
    world, row, column = wp.tid()
    if mask[world] != 0:
        destination[world, row, column] = source[world, row, column]


def copy_world_array(mask, source, destination):
    """Enqueue a masked per-world copy, including vec/matrix element arrays.

    Array shapes and dtypes must match and dimension zero must be the world
    dimension.  Packed contact/constraint arrays require separate treatment;
    this helper must not be used for them.
    """
    if source.shape != destination.shape or source.dtype != destination.dtype:
        raise ValueError("Masked copies require matching source/destination shapes and dtypes")
    if source.shape[0] != mask.shape[0]:
        raise ValueError("Masked copies require the world index as dimension zero")
    kernels = {1: masked_copy_1d, 2: masked_copy_2d, 3: masked_copy_3d}
    if source.ndim not in kernels:
        raise ValueError(f"Unsupported per-world array rank: {source.ndim}")
    if source.size:
        wp.launch(
            kernels[source.ndim],
            dim=source.shape,
            inputs=[mask, source, destination],
            device=source.device,
        )
