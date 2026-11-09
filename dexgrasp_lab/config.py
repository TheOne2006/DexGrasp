import json
from pathlib import Path

DEFAULTS = {
    "schema_version": 1, "architecture": "pointnet_mlp", "objective": "flow_matching",
    "width": 128, "depth": 2, "seed": 0, "device": "cuda:0", "batch_size": 8,
    "accumulation": 4, "max_steps": 10000, "lr": 0.0002, "weight_decay": 0.01,
    "grad_clip": 1.0, "num_points": 1024, "scale_m": 0.1, "val_every": 250,
    "save_every": 500, "val_objects": 32, "sample_steps": 5, "candidates": 10,
}


def load_config(path):
    raw = json.loads(Path(path).read_text())
    if set(raw) - set(DEFAULTS) - {"manifest", "run_dir", "checkpoint_dir"}:
        raise ValueError("unknown configuration keys")
    cfg = {**DEFAULTS, **raw}
    if cfg["schema_version"] != 1:
        raise ValueError("unsupported config version")
    if cfg["architecture"] not in ("pointnet_mlp", "pointnet_transformer"):
        raise ValueError("unknown architecture")
    if cfg["objective"] not in ("flow_matching", "diffusion"):
        raise ValueError("unknown objective")
    for k in ("width", "depth", "batch_size", "accumulation", "max_steps", "num_points",
              "val_every", "save_every", "val_objects", "sample_steps", "candidates"):
        if type(cfg[k]) is not int or cfg[k] < 1:
            raise ValueError(f"{k} must be a positive integer")
    if cfg["width"] < 8 or cfg["width"] % 8 or cfg["sample_steps"] > 1000:
        raise ValueError("invalid width or sample_steps")
    import math
    for k in ("lr", "scale_m", "grad_clip", "weight_decay"):
        if not isinstance(cfg[k], (float, int)) or not math.isfinite(cfg[k]) or cfg[k] < 0:
            raise ValueError(f"{k} must be finite and nonnegative")
        if k != "weight_decay" and cfg[k] == 0:
            raise ValueError(f"{k} must be positive")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("invalid seed")
    for k in ("manifest", "run_dir", "checkpoint_dir"):
        if not isinstance(cfg.get(k), str) or not cfg[k]:
            raise ValueError(f"missing {k}")
        cfg[k] = str((Path(path).resolve().parent / cfg[k]).resolve())
    return cfg
