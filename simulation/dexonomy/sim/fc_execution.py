"""Shared six-direction scheduling, checked GPU stepping and graph submission.

Protocols supply initial state, controls, force application, checkpoint timing,
and contact acceptance. This layer never resets a protocol or changes a verdict.
GPU imports are lazy so mapping/aggregation contracts remain CPU-only.
"""
DIRECTION_COUNT = 6


def direction_tasks(candidates):
    return [(candidate, direction) for candidate in candidates for direction in range(DIRECTION_COUNT)]


def add_direction(rows, direction, value):
    if direction not in range(DIRECTION_COUNT) or direction in rows:
        raise ValueError('unknown or duplicate candidate-direction world')
    rows[direction] = value


def ordered_directions(rows):
    if set(rows) != set(range(DIRECTION_COUNT)):
        raise ValueError('incomplete direction set')
    return [rows[d] for d in range(DIRECTION_COUNT)]


def checked_steps(model, data, count, active, errors, after_step=None, before_step=None):
    """Capture-safe common physical substep and persistent error history."""
    import mujoco_warp as mjw
    import warp as wp
    from dexonomy.sim.warp_stability_kernels import remember_errors
    for _ in range(count):
        if before_step is not None:
            before_step()
        mjw.step(model, data)
        wp.launch(remember_errors, active.shape[0], inputs=[active, data.overflow,
            int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS),
            data.nacon, data.nefc, data.naconmax, data.njmax, data.qpos,
            data.qvel, errors])
        if after_step is not None:
            after_step()


def capture_graph(submit):
    import warp as wp
    with wp.ScopedCapture() as capture:
        submit()
    return capture


def launch_graph(capture, count=1):
    import warp as wp
    for _ in range(count):
        wp.capture_launch(capture.graph)


def check_execution(data, errors, context, *, typed_nonfinite=False):
    """Preserve transient flags too; unknown/nonfinite/global failures fail closed."""
    import numpy as np
    import mujoco_warp as mjw
    from dexonomy.sim.mujoco_warp_env import check_warp_allocation, MuJoCoWarpAllocationError
    history = errors.numpy()
    try:
        diagnostics = check_warp_allocation(data, mjw.OverflowType, context=context)
    except MuJoCoWarpAllocationError as exc:
        diagnostics = dict(exc.diagnostics)
        diagnostics['allocation_overflow_flags'] = (np.asarray(diagnostics['allocation_overflow_flags']) | history).tolist()
        diagnostics['history_flags'] = history.tolist()
        raise MuJoCoWarpAllocationError(str(exc), diagnostics) from None
    if typed_nonfinite and np.any(history) and np.all((history == 0) | (history == 536870912)):
        from dexonomy.op.simulation_isolation import EvaluationNonfiniteError
        raise EvaluationNonfiniteError({**diagnostics, "history_flags": history.tolist()})
    if np.any(history):
        diagnostics['history_flags'] = history.tolist()
        diagnostics['allocation_overflow_flags'] = history.tolist()
        raise MuJoCoWarpAllocationError(context + ' recorded simulation error', diagnostics)
    return diagnostics
