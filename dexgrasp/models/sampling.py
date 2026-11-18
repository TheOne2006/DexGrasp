"""Conditional FM and FlowMap inference on encoded point clouds."""
from contextlib import contextmanager
import torch
from .keypoint import EncodedCondition


@contextmanager
def evaluation(model):
    previous = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(previous)


def repeat_context(context, candidates):
    if candidates == 1:
        return context
    def repeat(x):
        return None if x is None else x.repeat_interleave(candidates, dim=0)
    kv = None if context.kv is None else tuple(tuple(tuple(repeat(x) for x in branch)
                                                for branch in block) for block in context.kv)
    return EncodedCondition(tuple(map(repeat, context.tokens)), tuple(map(repeat, context.centers)),
                            repeat(context.gap_embedding), repeat(context.direction), kv,
                            repeat(context.direction_embedding))


@torch.no_grad()
def sample_keypoints(model, condition, *, steps, noise=None, cache_kv=True, candidates=1):
    if type(steps) is not int or steps < 1:
        raise ValueError('sample steps must be positive')
    if type(candidates) is not int or candidates < 1:
        raise ValueError('candidates must be positive')
    device = condition['object_points'].device
    shape = (len(condition['object_points'])*candidates, *model.cfg.state_shape)
    state = torch.randn(shape, device=device) if noise is None else noise.float().clone()
    if state.shape != shape or not torch.isfinite(state).all():
        raise ValueError('invalid initial noise')
    with evaluation(model):
        context = model.encode(condition, cache_kv=cache_kv)
        context = repeat_context(context, candidates)
        for i in range(steps):
            a = torch.full((len(state),), i/steps, device=device)
            if model.method == 'fm':
                state = state + model(state, a, context).float()/steps
            else:
                b = torch.full_like(a, (i+1)/steps)
                state = model(state, a, context, b).float()
        if not torch.isfinite(state).all():
            raise FloatingPointError('nonfinite generated keypoints')
        return model.cfg.absolute_state(state)
