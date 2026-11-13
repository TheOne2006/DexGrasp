"""Official d561ff9 Slide classes, unchanged; no force-closure preamble reuse."""
from dataclasses import dataclass
from typing import Any
import numpy as np
from dexonomy.util.np_util import np_interp_slide
from dexonomy.util.traj_util import MoveCfg, TrajectoryPlanner

@dataclass
class SlideCfg(MoveCfg):
    """Configuration for slide tasks."""

    axis: np.ndarray  # Direction vector for sliding movement, shape=(3,)
    distance: float  # Distance to slide along the axis


class SlidePlanner(TrajectoryPlanner):
    """Trajectory planner for slide tasks."""

    def __init__(self, move_cfg: SlideCfg, move_step: int = 10):
        super().__init__(move_cfg, move_step)

    def _interp_move_traj(self, init_pose: np.ndarray) -> np.ndarray:
        target_pose = np.copy(init_pose)
        target_pose[:3] += self.move_cfg.axis * self.move_cfg.distance
        return np_interp_slide(init_pose, target_pose, self.move_step)

    def _get_external_directions(self, move_obj_lst: np.ndarray) -> list[Any]:
        return [-self.move_cfg.axis] * len(move_obj_lst)
