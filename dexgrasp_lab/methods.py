"""Conditional straight-path FM and cosine-schedule epsilon diffusion/DDIM."""
import math
import torch

OBJECTIVES = ("flow_matching", "diffusion")


def alpha_bar(device, steps=1000):
    t = torch.linspace(0, 1, steps + 1, device=device, dtype=torch.float64)
    curve = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
    betas = (1 - curve[1:] / curve[:-1]).clamp(max=0.999)
    return (1 - betas).cumprod(0).float()


def training_pair(x, objective, *, noise=None, time=None):
    noise = torch.randn_like(x) if noise is None else noise
    shape = (-1,) + (1,) * (x.ndim - 1)
    if objective == "flow_matching":
        t = torch.rand(len(x), device=x.device) if time is None else time
        xt = (1 - t.reshape(shape)) * noise + t.reshape(shape) * x
        return xt, t, x - noise
    if objective == "diffusion":
        idx = torch.randint(0, 1000, (len(x),), device=x.device) if time is None else time
        a = alpha_bar(x.device)[idx].reshape(shape)
        return a.sqrt() * x + (1 - a).sqrt() * noise, idx.float() / 999, noise
    raise ValueError(f"unknown objective: {objective}")


def loss(model, point_cloud, x, objective):
    xt, t, target = training_pair(x, objective)
    pred = model(xt, t, model.encode(point_cloud))
    return (pred - target).square().mean()


@torch.no_grad()
def sample(model, point_cloud, objective, steps, candidates, *, noise=None):
    if objective not in OBJECTIVES or type(steps) is not int or not 1 <= steps <= 1000:
        raise ValueError("unknown objective or steps outside [1, 1000]")
    if type(candidates) is not int or candidates < 1:
        raise ValueError("candidates must be positive")
    context = model.encode(point_cloud).repeat_interleave(candidates, dim=0)
    shape = (len(context), 3, 28, 3)
    x = torch.randn(shape, device=point_cloud.device) if noise is None else noise.clone()
    if x.shape != shape:
        raise ValueError("wrong initial noise shape")
    if objective == "flow_matching":
        for i in range(steps):
            t = torch.full((len(x),), i / steps, device=x.device)
            x = x + model(x, t, context) / steps
    else:
        # DDIM eta=0, epsilon prediction; alpha_prev=1 at the clean endpoint.
        indices = torch.linspace(999, 0, steps, device=x.device).round().long()
        alphas = alpha_bar(x.device)
        for i, idx in enumerate(indices):
            a = alphas[idx]
            previous = alphas[indices[i + 1]] if i + 1 < steps else x.new_tensor(1.0)
            t = torch.full((len(x),), idx.item() / 999, device=x.device)
            eps = model(x, t, context)
            clean = (x - (1 - a).sqrt() * eps) / a.sqrt()
            x = previous.sqrt() * clean + (1 - previous).sqrt() * eps
    if not torch.isfinite(x).all():
        raise ValueError("nonfinite samples")
    return x.reshape(len(point_cloud), candidates, 3, 28, 3)
