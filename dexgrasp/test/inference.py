"""Inference on new raw scene points without a labelled training dataset."""
import torch
from ..train.objectives import sample_keypoints


@torch.no_grad()
def predict(model, raw_condition, *, candidates=1, steps=None, noise=None):
    """Batched raw inputs in metres; returns K[B,C,21,3] in the same scene frame.

    Required fields: object_points[B,1024,3], grasp_direction_cmd[B,3].
    full variants also require scene_points[B,2048,3], scene_mask[B,2048].
    Variants without the h_less suffix also require palm_gap[B,1] in metres.
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
    for key in (('object_points', 'scene_points') if model.cfg.use_scene else ('object_points',)):
        condition[key] = (condition[key]-center[:, None])/model.cfg.coord_scale_m
    if noise is not None:
        noise = torch.as_tensor(noise, dtype=torch.float32, device=device)
    steps = (10 if model.method == 'fm' else 1) if steps is None else steps
    generated = sample_keypoints(model, condition, steps=steps, candidates=candidates, noise=noise)
    return generated.reshape(len(points), candidates, 21, 3)*model.cfg.coord_scale_m+center[:, None, None]
