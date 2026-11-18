"""Construct commands for a frozen Sharpa pose using Weilun's resident contact and pregrasp code."""

import numpy as np


def resident_commands(host, entries, config, target):
    """Update commands in place and return per-candidate controller provenance.

    The incoming entries include a native CPU pregrasp for the hold fallback.
    Refinement starts at its exported grasp. No optimization step moves the prediction.
    """
    from dexonomy.sim.mujoco_warp_env import MuJoCoWarpAllocationError, release_warp_scene_memory
    from omegaconf import OmegaConf

    for capacity in (0, 2048, 4096, 8192):
        settings = OmegaConf.merge(config, dict(refine_nconmax=capacity, refine_nccdmax=capacity))
        try:
            return _resident_commands(host, entries, settings, target)
        except MuJoCoWarpAllocationError:
            if capacity == 8192:
                raise
        release_warp_scene_memory(str(config.warp_device))


def _resident_commands(host, entries, config, target):
    from dexonomy.op.gen_grasp_warp_resident import _contacts
    from dexonomy.qp.qp_single import ContactQP
    from dexonomy.sim.mujoco_warp_mixed import build_mixed_model
    from dexonomy.sim.mujoco_warp_resident_env import ResidentRefinementBatch

    if config.op.grasp.step != 0 or host._cfg.physics_profile != "native":
        raise ValueError("Frozen poses require native physics and zero grasp optimization steps")
    selected = []
    templates = []
    provenance = {}
    for entry in entries:
        data = entry["data"]
        q = data["grasp_qpos"]
        data["squeeze_qpos"] = q.copy()
        host.reset_qpos(q[0])
        contacts, _ = host.get_contacts(config.op.grasp.filter.contact.thre)
        if not len(contacts["pos"]):
            provenance[entry["id"]] = "hold_no_initial_contact"
            continue
        points = np.column_stack((contacts["pos"], contacts["normal"]))
        templates.append(
            dict(grasp_qpos=q, hand_cbody=contacts["bn1"], hand_cpn_w=points, obj_cpn_w=points)
        )
        selected.append(entry)
    if not selected:
        return provenance
    # The model bank captures the initial object and hand configuration, not a former world's final state.
    host.set_obj_margin(config.op.pregrasp.cdist)
    packed = build_mixed_model([host], [0] * len(selected), device=str(config.warp_device))
    batch = ResidentRefinementBatch(host, templates, config, mixed=packed)
    exported = batch.refine()
    contact_sets = []
    qp = config.op.grasp.filter.qp
    solver = ContactQP(qp.miu_coef)
    for i, entry in enumerate(selected):
        data = entry["data"]
        np.testing.assert_allclose(
            exported["qpos"][i, :29], data["grasp_qpos"][0], rtol=0, atol=2e-7
        )
        contacts, _ = _contacts(host, packed.original_geoms(exported["contacts"], i), 0)
        provenance[entry["id"]] = "hold_no_resident_contact"
        if len(contacts["pos"]):
            wrench, error = solver.solve(contacts["pos"], contacts["normal"], *target)
            provenance[entry["id"]] = "hold_qp_infeasible"
            if wrench is not None and np.isfinite(wrench).all() and error <= qp.thre:
                contacts["wrench"] = 10 * wrench
                provenance[entry["id"]] = "resident_qp"
                contact_sets.append(contacts)
                continue
        contact_sets.append(None)
    usable = [c is not None for c in contact_sets]
    if any(usable):
        result = batch.pregrasp(usable, contact_sets)
        for i, entry in enumerate(selected):
            if usable[i]:
                frames = int(result["iterations"][i])
                if frames <= 0:
                    raise ValueError(f"Empty pregrasp for {entry['id']}")
                data = entry["data"]
                data["pregrasp_qpos"] = result["trace"][i, :frames][::-2].copy()
                data["squeeze_qpos"] = result["squeeze"][i][None].copy()
    return provenance
