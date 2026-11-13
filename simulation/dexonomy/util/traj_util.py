from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass
import numpy as np

from dexonomy.util.np_util import (
    np_get_relative_pose,
    np_normalize_vector,
    np_interp_hinge,
    np_interp_slide,
    np_multiply_pose,
)


@dataclass
class MoveCfg:
    """Configuration for movement tasks."""

    type: str
    obj_name: str
    part_name: str



@dataclass
class ForceClosureCfg(MoveCfg):
    """Configuration for force closure tasks."""

    pass



class TrajectoryPlanner(ABC):
    """Abstract base class for trajectory planning strategies."""

    def __init__(
        self,
        move_cfg: ForceClosureCfg,
        move_step: int = 10,
    ):
        """
        Initialize trajectory planner.

        Args:
            move_cfg: Movement configuration dataclass
            move_step: Number of steps for movement interpolation
        """
        self.ctrl_qpos = []
        self.ext_fdir = []
        self.ctrl_type = []
        self.move_cfg = move_cfg
        self.move_step = move_step

    def plan_trajectory(
        self,
        init_obj_pose: np.ndarray,
        pregrasp_qpos: np.ndarray,
        grasp_qpos: np.ndarray,
        squeeze_qpos: np.ndarray,
        approach_qpos: np.ndarray | None,
    ) -> tuple[list[np.ndarray], list[str], list[np.ndarray], np.ndarray]:
        """
        Plan the trajectory for the specific task type.

        Returns:
            ctrl_qpos: list of qpos
            ctrl_type: list of control type (ee_pose, joint_angle)
            ext_fdir: list of external direction
            target_obj_pose: Target object pose after movement
        """
        if len(pregrasp_qpos.shape) == 1:
            pregrasp_qpos = pregrasp_qpos[None]
        if len(grasp_qpos.shape) == 1:
            grasp_qpos = grasp_qpos[None]
        if len(squeeze_qpos.shape) == 1:
            squeeze_qpos = squeeze_qpos[None]
        if approach_qpos is not None and len(approach_qpos.shape) == 1:
            approach_qpos = approach_qpos[None]
        self._add_approach_phase(approach_qpos)
        self._add_pregrasp_phase(pregrasp_qpos)
        self._add_grasp_phases(grasp_qpos, squeeze_qpos)

        # Check if movement is needed (for ForceClosure, no movement)
        if self._needs_movement():
            # Generate movement trajectory
            move_pose = self._interp_move_traj(squeeze_qpos[0, :7])
            move_qpos = np.repeat(squeeze_qpos, len(move_pose), axis=0)
            move_qpos[:, :7] = move_pose
            self.ctrl_qpos.extend(list(move_qpos))
            self.ctrl_type.extend(["ee_pose"] * move_qpos.shape[0])

            # Generate object movement trajectory
            move_obj_lst = self._interp_move_traj(init_obj_pose)
            if move_obj_lst is not None and len(move_obj_lst) > 0:
                target_obj_pose = move_obj_lst[-1]
            else:
                target_obj_pose = init_obj_pose

            # Add task-specific external directions
            self.ext_fdir.extend(self._get_external_directions(move_obj_lst))

            # Fill None values in ext_fdir
            self._fill_none_ext_fdir()
        else:
            # No movement needed (e.g., ForceClosure)
            target_obj_pose = init_obj_pose

        return self.ctrl_qpos, self.ctrl_type, self.ext_fdir, target_obj_pose

    def _add_approach_phase(self, approach_qpos: np.ndarray | None):
        """Add approach phase to trajectory if provided."""
        if approach_qpos is not None:
            self.ctrl_qpos.extend(list(approach_qpos))
            n_approach = approach_qpos.shape[0] - 1
            self.ext_fdir.extend([None] * n_approach)
            self.ctrl_type.extend(["joint_angle"] * n_approach)

    def _add_pregrasp_phase(self, pregrasp_qpos: np.ndarray):
        """Add pregrasp phase to trajectory."""
        self.ext_fdir.extend([None] * pregrasp_qpos.shape[0])
        self.ctrl_type.extend(["ee_pose"] * pregrasp_qpos.shape[0])
        self.ctrl_qpos.extend(list(pregrasp_qpos))

    def _add_grasp_phases(self, grasp_qpos: np.ndarray, squeeze_qpos: np.ndarray):
        """Add grasp and squeeze phases to trajectory."""
        # Grasp phase
        self.ext_fdir.extend([None])
        self.ctrl_type.extend(["ee_pose"])
        self.ctrl_qpos.extend(list(grasp_qpos))

        # Squeeze phase
        self.ext_fdir.extend([None])
        self.ctrl_type.extend(["ee_pose"])
        self.ctrl_qpos.extend(list(squeeze_qpos))

    def _fill_none_ext_fdir(self):
        """Fill None values in ext_fdir with the next non-None value."""
        i = len(self.ext_fdir) - 1
        for ext_fdir in reversed(self.ext_fdir):
            if ext_fdir is None:
                self.ext_fdir[i] = self.ext_fdir[i + 1]
            i -= 1

    def _needs_movement(self) -> bool:
        """Check if the task type requires movement after grasp."""
        return not isinstance(self.move_cfg, ForceClosureCfg)

    @abstractmethod
    def _interp_move_traj(self, init_pose: np.ndarray) -> np.ndarray:
        """Interpolate movement trajectory for specific task type."""
        pass

    @abstractmethod
    def _get_external_directions(self, move_obj_lst: np.ndarray) -> list[Any]:
        """Get external directions for the specific task type."""
        pass



class ForceClosurePlanner(TrajectoryPlanner):
    """Trajectory planner for force closure tasks."""

    def __init__(self, move_cfg: ForceClosureCfg, move_step: int = 10):
        super().__init__(move_cfg, move_step)

    def _interp_move_traj(self, init_pose: np.ndarray) -> np.ndarray:
        """No movement for force closure."""
        return np.array([])

    def _get_external_directions(self, move_obj_lst: np.ndarray) -> list[Any]:
        return []



def get_planner(move_cfg: dict[str, Any], move_step: int = 10) -> TrajectoryPlanner:
    """Get the appropriate planner for the given move configuration."""
    task_type = move_cfg["type"]
    move_cfg["part_name"] = move_cfg.get("part_name", None)
    move_cfg = TASK_TYPE_TO_CFG[task_type](**move_cfg)
    planner = TASK_TYPE_TO_PLANNER[task_type](move_cfg, move_step)
    return planner


TASK_TYPE_TO_CFG = {"force_closure": ForceClosureCfg}
TASK_TYPE_TO_PLANNER = {"force_closure": ForceClosurePlanner}
