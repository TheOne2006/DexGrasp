"""Straight-path FM and finite-interval distillation of frozen teacher ODE trajectories."""
import torch
from ..models.sampling import evaluation
from ..models.registry import sample_model as sample_keypoints  # Historical sampling API.


def fm_pair(clean, *, noise=None, time=None):
    noise = torch.randn_like(clean) if noise is None else noise
    generated_time = time is None
    if generated_time:
        time = torch.rand(len(clean), device=clean.device)
    if noise.shape != clean.shape or time.shape != (len(clean),):
        raise ValueError('FM noise/time shape mismatch')
    if not generated_time and not ((time >= 0) & (time <= 1)).all():
        raise ValueError('FM time outside [0,1]')
    weight = time.view(-1, *([1] * (clean.ndim - 1)))
    return (1-weight)*noise + weight*clean, time, clean-noise


def fm_loss(model, batch):
    if model.method != 'fm':
        raise ValueError('fm_loss requires an FM model')
    if hasattr(model, 'training_loss'):
        return model.training_loss(batch)
    clean = batch['keypoints_gt']
    if hasattr(model.cfg, 'training_state'):
        clean = model.cfg.training_state(clean)
    state, time, target = fm_pair(clean)
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
    noise = torch.randn(len(batch['keypoints_gt']), *student.cfg.state_shape, device=batch['keypoints_gt'].device)
    trajectory = ode_trajectory(teacher, batch['condition'], noise, steps=steps, solver=solver)
    state_a, state_b, a, b = interval_pairs(trajectory, full_interval_probability)
    predicted = student(state_a, a, student.encode(batch['condition']), b)
    return (predicted.float()-state_b.float()).square().mean()
