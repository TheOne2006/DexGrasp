"""Adapter from a single Sharpa grasp to the original Dexonomy three-stage controls."""

from pathlib import Path
import json

import numpy as np


def controls(opt, grasp, settings, *, qp_target=None):
    """Derive commands from the predicted pose, using upstream contact QP and pregrasp rollout."""
    from dexonomy.op.gen_grasp import GraspFilter
    from dexonomy.qp.qp_single import ContactQP

    opt.set_obj_margin(settings.grasp.cdist)
    opt.reset_qpos(grasp)
    contact, _ = opt.get_contacts(settings.grasp.filter.contact.thre)
    squeeze = grasp.copy()
    source = "hold_no_contact"
    if len(contact["pos"]):
        qp_cfg = settings.grasp.filter.qp
        if qp_target is None:
            ext_wrench, ext_center = np.zeros(6), opt.get_active_obj_pose()[:3]
        else:
            ext_wrench, ext_center = qp_target
        wrench, error = ContactQP(qp_cfg.miu_coef).solve(
            contact["pos"], contact["normal"], ext_wrench, ext_center
        )
        if wrench is not None and np.isfinite(wrench).all() and error <= qp_cfg.thre:
            squeeze = opt.get_squeeze_qpos(grasp, contact["bn1"], contact["pos"], 10 * wrench)
            source = "contact_qp"
        else:
            source = "hold_qp_infeasible"
    config = settings.pregrasp
    filtered = GraspFilter(config.filter, opt, "predicted grasp")
    trajectory = []
    for step in range(config.step):
        margin = config.cdist * min((step + 1) * 2 / config.step, 1)
        opt.set_obj_margin(margin)
        opt.keep_hand_stable()
        opt.step_sim(settings.grasp.substep)
        trajectory.append(opt.get_hand_qpos().copy())
        if step % 2 == 0 and margin >= config.cdist and filtered.early_stop():
            break
    return np.stack(trajectory[::-2]), squeeze[None], source


def floating_scene(assets, geometry):
    """Build an object-frame force-closure scene from a prepared collision asset."""
    folder = Path(assets) / geometry["asset_sha256"]
    target = dict(
        type="rigid_object",
        xml_path=str(folder / "coacd.xml"),
        info_path=str(folder / "info.json"),
        scale=np.asarray(geometry["scale"]),
        pose=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )
    return dict(scene={"target": target}, task=dict(obj_name="target", type="force_closure"))


def make_scene(scene, hand_xml, spec, settings):
    from dexonomy.sim import HandCfg, MuJoCo_OptCfg, MuJoCo_OptEnv, MuJoCo_EvalCfg, MuJoCo_EvalEnv

    obj = scene["scene"][scene["task"]["obj_name"]]
    info = json.loads(Path(obj["info_path"]).read_text())
    opt = MuJoCo_OptEnv(
        HandCfg(str(hand_xml), freejoint=True),
        scene,
        MuJoCo_OptCfg(obj_margin=settings.pregrasp.cdist),
    )
    obj["density"] = 0.1 / (
        info["mass"] / (info["density"] * info["scale"] ** 3) * np.prod(obj["scale"])
    )
    env = MuJoCo_EvalEnv(
        HandCfg(str(hand_xml), freejoint=True),
        scene,
        MuJoCo_EvalCfg(miu_coef=[0.6, 0.02]),
        debug_view=False,
    )
    target = env.get_active_obj_pose().copy()
    names = [
        opt._model.joint(j).name.removeprefix(opt._cfg.hand_prefix)
        for j in range(opt._model.njnt)
        if 7 <= opt._model.jnt_qposadr[j] < 29
    ]
    assert set(names) == set(spec["joint_order"]) and len(names) == 22
    order = [spec["joint_order"].index(name) for name in names]
    return scene, opt, env, target, order


def test_pose(context, pose, frame, settings, paired=False, bench=None):
    from scipy.spatial.transform import Rotation
    from dexonomy.util.traj_util import get_planner

    scene, opt, env, target, order = context
    quat = Rotation.from_matrix(pose["R"] @ frame.T).as_quat(scalar_first=True)
    grasp = np.r_[pose["t"], quat, pose["q"][order]][None]
    pre, squeeze, source = controls(opt, grasp[0], settings)
    arms = {"derived": (pre, grasp, squeeze)}
    if paired:
        arms.update(no_squeeze=(pre, grasp, grasp), hold=(grasp, grasp, grasp))
    outcomes = {}
    for name, stages in arms.items():
        trajectory = get_planner(scene["task"]).plan_trajectory(target.copy(), *stages, None)
        passed, _, _ = env.eval_fc(*trajectory, 0.05, 15.0)
        warned = env._data.warning.number.any() or not np.isfinite(env._data.qpos).all()
        outcomes[name] = bool(passed and not warned)
    if bench is not None:
        from .bench_force import evaluate

        outcomes["bench"] = evaluate(bench, pre[0], grasp[0], squeeze[-1])
    return outcomes, source
