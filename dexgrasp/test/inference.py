"""Inference on new raw scene points without a labelled training dataset."""
import torch
from ..models.registry import sample_model


@torch.no_grad()
def predict(model, raw_condition, *, candidates=1, steps=None, noise=None):
    """Return metric K[B,C,21,3] (v1) or K[B,C,3,21,3] (pregrasp, grasp, squeeze).

    All models require object_points[B,1024,3]. Direction branches require grasp_direction_cmd[B,3].
    full variants also require scene_points[B,2048,3], scene_mask[B,2048].
    Depth-conditioned models require palm_gap (v1) or grasp-frame h32 (v2), [B,1] in metres.
    wrist-z-v2 checkpoints require scene-frame official +Zh (wrist toward fingers);
    legacy checkpoints retain their original palm-normal condition.
    This function does not infer visibility, segment targets or perform physical checks.
    """
    device = next(model.parameters()).device
    condition = {k: torch.as_tensor(raw_condition[k], device=device, dtype=torch.float32)
                 for k in model.cfg.condition_keys}
    # Validate before normalizing; encoding later validates mask counts and direction norm.
    points = condition['object_points']
    if points.ndim != 3 or points.shape[1:] != (1024, 3):
        raise ValueError('raw object_points must be [B,1024,3]')
    # Accumulate the center in float64, matching the dataset's NumPy normalization.
    center = points.double().mean(1).float()
    for key in ('object_points', 'scene_points'):
        if key in condition:
            condition[key] = (condition[key]-center[:, None])/model.cfg.coord_scale_m
    if noise is not None:
        noise = torch.as_tensor(noise, dtype=torch.float32, device=device)
    generated = sample_model(model, condition, steps=steps, candidates=candidates, noise=noise)
    return world_keypoints(generated, center, model.cfg.coord_scale_m, candidates)


def world_keypoints(generated, center, scale, candidates):
    poses = generated.unflatten(0, (len(center), candidates))
    origin = center.view(len(center), *([1] * (poses.ndim - 2)), 3)
    return poses * scale + origin
