"""Straight-path FM and finite-interval distillation of frozen teacher ODE trajectories."""
from contextlib import contextmanager
import torch


@contextmanager
def evaluation(model):
    previous = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(previous)


def fm_pair(clean, *, noise=None, time=None):
    noise = torch.randn_like(clean) if noise is None else noise
    time = torch.rand(len(clean), device=clean.device) if time is None else time
    if noise.shape != clean.shape or time.shape != (len(clean),):
        raise ValueError('FM noise/time shape mismatch')
    if not torch.isfinite(time).all() or ((time < 0) | (time > 1)).any():
        raise ValueError('FM time outside [0,1]')
    weight = time[:, None, None]
    return (1-weight)*noise + weight*clean, time, clean-noise


def fm_loss(model, batch):
    if model.method != 'fm':
        raise ValueError('fm_loss requires an FM model')
    state, time, target = fm_pair(batch['keypoints_gt'])
    prediction = model(state, time, model.encode(batch['condition']))
    return (prediction.float()-target.float()).square().mean()


@torch.no_grad()
def ode_trajectory(model, condition, noise, *, steps, solver='euler', cache_kv=True):
    if model.method != 'fm' or type(steps) is not int or steps < 1 or solver not in ('euler', 'heun'):
        raise ValueError('FM teacher, positive steps and euler/heun solver required')
    with evaluation(model):
        context = model.encode(condition, cache_kv=cache_kv)
        state = noise.float().clone()
        states = [state]
        for i in range(steps):
            start = torch.full((len(state),), i/steps, device=state.device)
            velocity = model(state, start, context).float()
            proposed = state + velocity/steps
            if solver == 'heun':
                end = torch.full_like(start, (i+1)/steps)
                velocity_end = model(proposed, end, context).float()
                proposed = state + (velocity+velocity_end)/(2*steps)
            state = proposed
            states.append(state)
        trajectory = torch.stack(states, 1)
        if not torch.isfinite(trajectory).all():
            raise FloatingPointError('teacher produced nonfinite trajectory')
        return trajectory


def interval_pairs(trajectory, full_interval_probability=0.5):
    """Both endpoints always come from one trajectory/condition; includes exact (0,1)."""
    b, time_count = trajectory.shape[:2]
    steps = time_count-1
    if steps < 1 or not 0 < full_interval_probability <= 1:
        raise ValueError('invalid trajectory or full-interval probability')
    # Uniform choice among unordered distinct grid-index pairs.
    a = torch.randint(0, time_count, (b,), device=trajectory.device)
    other = torch.randint(0, steps, (b,), device=trajectory.device)
    other = other + (other >= a)
    start, end = torch.minimum(a, other), torch.maximum(a, other)
    full = torch.rand(b, device=trajectory.device) < full_interval_probability
    start = torch.where(full, 0, start)
    end = torch.where(full, steps, end)
    rows = torch.arange(b, device=trajectory.device)
    return trajectory[rows, start].detach(), trajectory[rows, end].detach(), start.float()/steps, end.float()/steps


def flowmap_loss(student, teacher, batch, *, steps=32, solver='heun', full_interval_probability=0.5):
    if student.method != 'flowmap' or teacher.method != 'fm' or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('FlowMap requires a student and fully frozen FM teacher')
    # No dependence on the paired ground-truth endpoint: supervision is the actual teacher trajectory.
    noise = torch.randn_like(batch['keypoints_gt'])
    trajectory = ode_trajectory(teacher, batch['condition'], noise, steps=steps, solver=solver)
    state_a, state_b, a, b = interval_pairs(trajectory, full_interval_probability)
    predicted = student(state_a, a, student.encode(batch['condition']), b)
    return (predicted.float()-state_b.float()).square().mean()


@torch.no_grad()
def sample_keypoints(model, condition, *, steps, noise=None, cache_kv=True, candidates=1):
    if type(steps) is not int or steps < 1:
        raise ValueError('sample steps must be positive')
    if type(candidates) is not int or candidates < 1:
        raise ValueError('candidates must be positive')
    device = condition['object_points'].device
    shape = (len(condition['object_points'])*candidates, model.cfg.keypoints, 3)
    state = torch.randn(shape, device=device) if noise is None else noise.float().clone()
    if state.shape != shape or not torch.isfinite(state).all():
        raise ValueError('invalid initial noise')
    with evaluation(model):
        context = model.encode(condition, cache_kv=cache_kv)
        if candidates > 1:
            from ..models.keypoint import EncodedCondition
            repeat = lambda x: x.repeat_interleave(candidates, dim=0)
            kv = None if context.kv is None else tuple(tuple(tuple(repeat(x) for x in branch)
                                                               for branch in block) for block in context.kv)
            context = EncodedCondition(tuple(map(repeat, context.tokens)), tuple(map(repeat, context.centers)),
                                       repeat(context.gap_embedding), repeat(context.direction), kv)
        for i in range(steps):
            a = torch.full((len(state),), i/steps, device=device)
            if model.method == 'fm':
                state = state + model(state, a, context).float()/steps
            else:
                b = torch.full_like(a, (i+1)/steps)
                state = model(state, a, context, b).float()
        if not torch.isfinite(state).all():
            raise FloatingPointError('nonfinite generated keypoints')
        return state
