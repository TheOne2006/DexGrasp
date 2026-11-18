"""Replay three-stage poses with the fc-mocap implementation supplied by DexGraspBench."""

from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec
import sys
from pathlib import Path

import numpy as np


def load_backend(checkout):
    """Import the user-specified checkout in a benchmark-only process."""
    source = Path(checkout).resolve() / "src"
    sys.path.insert(0, str(source))
    # Upstream task/__init__ imports its USD viewer and dataset statistics unconditionally.
    # Load the physical evaluator as a package from the same checkout, with no viewer setup.
    if "task" not in sys.modules:
        package = ModuleSpec("task", loader=None, is_package=True)
        package.submodule_search_locations = [str(source / "task")]
        sys.modules["task"] = module_from_spec(package)
    from task.eval_func.fc_mocap import fcMocapEval
    from util.hand_util import MjHO

    return fcMocapEval, MjHO


def build_replay(backend, object_dir, scale, density, hand_xml, config, obj_pose):
    """Compile one official simulator per scene and reuse it across candidates."""
    evaluator, model = backend
    replay = evaluator.__new__(evaluator)
    replay.mj_ho = model(
        obj_path=str(object_dir),
        obj_scale=float(scale),
        obj_density=float(density),
        hand_xml_path=str(hand_xml),
        hand_mocap=True,
        exclude_table_contact=None,
        friction_coef=config.task.miu_coef,
        has_floor_z0=False,
    )
    replay.configs = config
    replay.grasp_data = {"obj_pose": np.array(obj_pose, dtype=float)}
    return replay


def evaluate(replay, pre, grasp, squeeze):
    """The upstream evaluator resets each direction, checks penetration, then tests motion."""
    replay.grasp_data.update(pregrasp_qpos=pre, grasp_qpos=grasp, squeeze_qpos=squeeze)
    passed, _, _ = replay._eval_simulate_under_extforce()
    data = replay.mj_ho.data
    return bool(passed and not data.warning.number.any() and np.isfinite(data.qpos).all())


def load_config(checkout):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "setting": "fc",
            "hand": {"mocap": True},
            "task": OmegaConf.load(Path(checkout) / "config/task/eval.yaml"),
        }
    )
