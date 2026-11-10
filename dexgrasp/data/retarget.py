"""Shadow-to-Sharpa landmark fitting for kinematic diagnostics, not grasp validation.

Input convention: Dexonomy root xyz + quaternion wxyz + XML-order Shadow joints.
The upstream XML (and its license) must be retained in the source manifest.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

SHADOW_FROM_SHARPA = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
TIP_INDICES = np.array([4, 8, 12, 16, 20])


class ShadowLandmarks:
    def __init__(self, xml):
        import mujoco
        self.mj = mujoco
        root = ET.parse(xml).getroot()
        # Kinematics only: retain joint defaults, body transforms and inertias.
        for child in list(root):
            if child.tag not in ('compiler', 'default', 'worldbody'):
                root.remove(child)
        def clean(element):
            for child in list(element):
                if child.tag not in ('compiler', 'default', 'worldbody', 'body', 'joint', 'inertial'):
                    element.remove(child)
                else:
                    clean(child)
        clean(root)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        self.joint_order = [self.model.joint(i).name for i in range(self.model.njnt)]
        if self.model.nq != 22:
            raise ValueError('expected exactly 22 Shadow finger coordinates')
        self.points = [('rh_palm', [0, 0, 0]),
                       ('rh_thbase', [0, 0, 0]), ('rh_thhub', [0, 0, 0]),
                       ('rh_thdistal', [0, 0, 0]), ('rh_thdistal', [0, 0, .03])]
        for finger in ('ff', 'mf', 'rf', 'lf'):
            for link in ('proximal', 'middle', 'distal'):
                self.points.append((f'rh_{finger}{link}', [0, 0, 0]))
            self.points.append((f'rh_{finger}distal', [0, 0, .03]))
        self.ids = [self.model.body(name).id for name, _ in self.points]

    def local(self, q):
        self.data.qpos[:] = q
        self.mj.mj_kinematics(self.model, self.data)
        return np.array([self.data.xpos[i] + self.data.xmat[i].reshape(3, 3) @ p
                         for i, (_, p) in zip(self.ids, self.points)])

    def world(self, qpos):
        qpos = np.asarray(qpos, float)
        if qpos.shape != (29,) or not np.isfinite(qpos).all():
            raise ValueError('invalid Shadow pose')
        if not np.isclose(np.linalg.norm(qpos[3:7]), 1., atol=1e-4):
            raise ValueError('non-unit source quaternion')
        rotation = Rotation.from_quat(qpos[3:7], scalar_first=True).as_matrix()
        return self.local(qpos[7:]) @ rotation.T + qpos[:3]


def retarget(hand, shadow, source_qpos):
    """Keep source palm orientation; fit translation (±3 cm) and bounded Sharpa q.

    Residual: anatomical landmarks, fingertip weight 2, weak translation prior.
    No copying joint angles between the two hands, mesh deformation or label scaling.
    """
    source = shadow.world(source_qpos)
    R = Rotation.from_quat(source_qpos[3:7], scalar_first=True).as_matrix() @ SHADOW_FROM_SHARPA
    origin = source[0]
    target = (source-origin) @ R
    weights = np.ones((21, 1)); weights[TIP_INDICES] = 2.
    q0 = np.clip(np.full(22, .2), hand.lower+1e-6, hand.upper-1e-6)
    last_x, cache = None, None
    def compute(x):
        nonlocal last_x, cache
        if last_x is None or not np.array_equal(last_x, x):
            p, j = hand.fk(x[3:], jacobian=True)
            residual = np.concatenate((((p+x[:3]-target)*weights).ravel(), .25*x[:3]))
            jac = np.concatenate((np.broadcast_to(np.eye(3), (21, 3, 3)), j), axis=2)
            jac = np.concatenate(((jac*weights[:, :, None]).reshape(63, 25),
                                  np.pad(.25*np.eye(3), ((0, 0), (0, 22)))), axis=0)
            last_x, cache = x.copy(), (residual, jac)
        return cache
    fit = least_squares(lambda x: compute(x)[0], np.r_[np.zeros(3), q0],
                        jac=lambda x: compute(x)[1],
                        bounds=(np.r_[[-.03]*3, hand.lower], np.r_[[.03]*3, hand.upper]),
                        max_nfev=80, ftol=1e-6, xtol=1e-6, gtol=1e-6)
    t, q = origin+R@fit.x[:3], fit.x[3:]
    points = hand.world_keypoints(R, t, q)
    errors = np.linalg.norm(points-source, axis=-1)
    return dict(R=R, t=t, q=q, K=points, shadow_K=source,
                mpjpe_m=float(errors.mean()), fingertip_max_m=float(errors[TIP_INDICES].max()),
                optimizer_success=bool(fit.success), nfev=int(fit.nfev))
