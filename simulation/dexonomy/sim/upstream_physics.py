"""The effective physics settings of Dexonomy's pinned MuJoCo 3.3.3.

Upstream commit d561ff9df8545deef7a3f9e7cfb4045ef66f6913 pins MuJoCo
3.3.3.  Its source leaves several settings at that version's defaults.  This
module restores the identified defaults and call semantics on MuJoCo 3.17;
it does not claim that a different engine has identical collision numerics.
"""

from math import sqrt

import mujoco


UPSTREAM_PHYSICS_PROFILE = "upstream_mujoco333"
NATIVE_PHYSICS_PROFILE = "native"
UPSTREAM_MUJOCO_VERSION = "3.3.3"
UPSTREAM_COMMIT = "d561ff9df8545deef7a3f9e7cfb4045ef66f6913"


def needs_legacy_forward_warmstart(profile: str) -> bool:
    """Whether standalone forward needs the 3.3.3 warmstart update.

    Native 3.3.3 already performs this update and must remain an exact no-op.
    The compatibility implementation is audited for 3.17.0, including Warp's
    matching forward/step distinction.  Reject unreviewed engine versions.
    """
    if profile not in (UPSTREAM_PHYSICS_PROFILE, NATIVE_PHYSICS_PROFILE):
        raise ValueError(f"Unknown physics_profile: {profile!r}")
    if profile == NATIVE_PHYSICS_PROFILE or mujoco.__version__ == UPSTREAM_MUJOCO_VERSION:
        return False
    if mujoco.__version__ != "3.17.0":
        raise ValueError(
            f"{UPSTREAM_PHYSICS_PROFILE} is audited for MuJoCo 3.3.3 and 3.17.0; "
            f"got {mujoco.__version__}. Use physics_profile='native' explicitly "
            "to use another engine's own defaults."
        )
    return True


def configure_physics_options(spec, profile: str) -> None:
    """Restore changed 3.3.3 defaults before compilation; preserve stage options."""
    if not needs_legacy_forward_warmstart(profile):
        return
    # Flags moved from opt-in enable bits to opt-out disable bits.  Never copy
    # their raw integers across MuJoCo versions (even GRAVITY changed bit index).
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ISLAND)
    spec.option.ccd_iterations = 50


def restore_upstream_actuator_damping(spec, model, profile: str) -> None:
    """Reproduce 3.3.3's dampratio compilation for any single-output actuator.

    MuJoCo 3.3.3 engine_setconst.c:342-375 uses a diagonal reflected inertia:
      mass = sum(dof_M0[j] / transmission[j]**2)
      kv = 2 * dampratio * sqrt(kp * mass)
    3.17 uses 1 / (J inv(M) J'), which changes kv even for unchanged XML.
    Read the original ratio from MjSpec, never infer it from compiled kv or
    hard-code a particular hand. Explicit kv (nonpositive biasprm[2]) stays as is.
    """
    if not needs_legacy_forward_warmstart(profile):
        return
    actuators = list(spec.actuators)
    changed = [
        (index, actuator) for index, actuator in enumerate(actuators)
        if actuator.gainprm[0] == -actuator.biasprm[1] and actuator.biasprm[2] > 0
    ]
    if not changed:
        return
    if len(actuators) != getattr(model, "nactuator", model.nu):
        raise ValueError("Cannot map specification actuators to compiled actuators")
    # Separate data at qpos0: obtaining the transmission must not mutate the
    # simulation's reset state, warmstart, contacts, or control matrix.
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_tendon(model, data)
    mujoco.mj_transmission(model, data)
    for index, actuator in changed:
        actuator_id = model.actuator(actuator.name).id if actuator.name else index
        row = actuator_id
        if hasattr(model, "actuator_outnum"):
            if int(model.actuator_outnum[actuator_id]) != 1:
                raise ValueError("MuJoCo 3.3.3 has no multi-output actuator damping semantics")
            row = int(model.actuator_outadr[actuator_id])
        address = int(data.moment_rowadr[row])
        count = int(data.moment_rownnz[row])
        mass = 0.0
        for offset in range(address, address + count):
            transmission = abs(float(data.actuator_moment[offset]))
            squared = transmission * transmission
            if squared > mujoco.mjMINVAL:
                dof = int(data.moment_colind[offset])
                mass += float(model.dof_M0[dof]) / squared
        kp, ratio = float(actuator.gainprm[0]), float(actuator.biasprm[2])
        model.actuator_biasprm[actuator_id, 2] = -2 * ratio * sqrt(kp * mass)


def forward_with_physics_profile(model, data, profile: str) -> None:
    """Execute the upstream standalone-forward semantics on the native CPU."""
    update_warmstart = needs_legacy_forward_warmstart(profile)
    mujoco.mj_forward(model, data)
    if update_warmstart:
        # 3.3.6 moved this copy from forward dynamics to time integration.  The
        # upstream algorithm also uses standalone forward for reset and contact
        # queries, so those calls need the former update explicitly restored.
        data.qacc_warmstart[:] = data.qacc
