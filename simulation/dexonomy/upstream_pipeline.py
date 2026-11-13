"""Stable-grasp validation shared by the streaming pipeline."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np

class PipelineError(ValueError):
    """Raised when an upstream pipeline stage is used out of order."""



def _load_dict(path: Path) -> dict[str, Any]:
    try:
        value = np.load(path, allow_pickle=True).item()
    except Exception as exc:
        raise PipelineError(f"cannot load Dexonomy data {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"Dexonomy data is not a dictionary: {path}")
    return value



def _stable_score(data: dict[str, Any], path: Path) -> float:
    metrics = data.get("stability_metrics")
    if not isinstance(metrics, dict):
        raise PipelineError(
            f"succ_grasp lacks six-direction stability metrics: {path}; "
            "rerun force-closure evaluation with the corrected pipeline"
        )
    if metrics.get("type") != "force_closure_6d" or not metrics.get("passes", False):
        raise PipelineError(f"not a force-closure-approved succ_grasp: {path}")
    if metrics.get("directions_completed") != metrics.get("directions_required", 6):
        raise PipelineError(f"six-direction evaluation is incomplete: {path}")
    score = float(metrics.get("stability_score", float("inf")))
    if not np.isfinite(score):
        raise PipelineError(f"invalid stability score in {path}")
    return score
