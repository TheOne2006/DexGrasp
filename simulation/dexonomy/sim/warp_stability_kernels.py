"""Per-world clocks for the unchanged six-direction force-closure test."""

import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.kernel
def prepare_round(
    clock: wp.array[int],
    preamble_rounds: wp.array[int],
    controls: wp.array3d[float],
    mocap_positions: wp.array2d[wp.vec3],
    mocap_quaternions: wp.array2d[wp.quat],
    active: wp.array[int],
    ctrl: wp.array2d[float],
    mocap_pos: wp.array2d[wp.vec3],
    mocap_quat: wp.array2d[wp.quat],
):
    world = wp.tid()
    step = clock[0]
    active[world] = int(step < preamble_rounds[world] + 100)
    if step < preamble_rounds[world]:
        for actuator in range(ctrl.shape[1]):
            ctrl[world, actuator] = controls[world, step, actuator]
        mocap_pos[world, 0] = mocap_positions[world, step]
        mocap_quat[world, 0] = mocap_quaternions[world, step]


@wp.kernel
def start_external_force(
    clock: wp.array[int],
    preamble_rounds: wp.array[int],
    object_body: wp.array[int],
    forces: wp.array[wp.vec3],
    subtree_com: wp.array2d[wp.vec3],
    xipos: wp.array2d[wp.vec3],
    xfrc: wp.array2d[wp.spatial_vector],
):
    """The force is mapped once using the final preamble's derived state."""
    world = wp.tid()
    if clock[0] == preamble_rounds[world]:
        body = object_body[world]
        force = forces[world]
        torque = wp.cross(subtree_com[world, body] - xipos[world, body], force)
        xfrc[world, body] = wp.spatial_vector(force, torque)


@wp.kernel
def record_checkpoint(
    clock: wp.array[int],
    preamble_rounds: wp.array[int],
    object_body: wp.array[int],
    xpos: wp.array2d[wp.vec3],
    xquat: wp.array2d[wp.quat],
    checkpoint_pos: wp.array2d[wp.vec3],
    checkpoint_quat: wp.array2d[wp.quat],
    checkpoint_seen: wp.array2d[int],
    active: wp.array[int],
    newly_complete: wp.array[int],
    completed: wp.array[int],
):
    world = wp.tid()
    elapsed = clock[0] + 1 - preamble_rounds[world]
    newly_complete[world] = int(elapsed == 100)
    completed[world] = int(elapsed > 100)
    if active[world] != 0 and elapsed > 0 and elapsed <= 100 and elapsed % 10 == 0:
        hold = elapsed // 10 - 1
        body = object_body[world]
        checkpoint_pos[world, hold] = xpos[world, body]
        checkpoint_quat[world, hold] = xquat[world, body]
        checkpoint_seen[world, hold] = 1


@wp.kernel
def advance_clock(clock: wp.array[int]):
    clock[0] += 1


@wp.kernel
def remember_errors(
    active: wp.array[int],
    flags: wp.array[int],
    solver_mask: int,
    nacon: wp.array[int],
    nefc: wp.array[int],
    contact_capacity: int,
    constraint_capacity: int,
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    history: wp.array[int],
):
    world = wp.tid()
    if active[world] != 0:
        value = flags[world] & ~solver_mask
        if nacon[0] > contact_capacity or nefc[world] > constraint_capacity:
            value = value | 1073741824
        for index in range(qpos.shape[1]):
            if not wp.isfinite(qpos[world, index]):
                value = value | 536870912
        for index in range(qvel.shape[1]):
            if not wp.isfinite(qvel[world, index]):
                value = value | 536870912
        history[world] = history[world] | value


@wp.kernel
def apply_retirement(retired: wp.array[int], active: wp.array[int]):
    world = wp.tid()
    if retired[world] != 0:
        active[world] = 0


@wp.kernel
def accumulate_solver_flags(active: wp.array[int], flags: wp.array[int],
                            solver_mask: int, history: wp.array[int]):
    world = wp.tid()
    if active[world] != 0:
        history[world] = history[world] | (flags[world] & solver_mask)
