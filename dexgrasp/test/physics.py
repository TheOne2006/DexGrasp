"""Sharpa single-pose evaluation inspired by KPGrasp/DexGraspBench.

Independent implementation of the published metric definitions; no upstream
code is copied. This is not the Shadow three-stage benchmark. The native
Sharpa position controllers hold the decoded pose, with a fixed wrist.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
PROTOCOL = dict(
    schema='sharpa-single-pose-force-test-v1', timestep_s=.004,
    arena_memory_mib=128,
    object_mass_kg=.1, friction=[.6, .02, .001], gravity=[0, 0, 0],
    force_acceleration_m_s2=10., force_directions=np.vstack((np.eye(3), -np.eye(3))).tolist(),
    settling_steps=250, force_steps=500, check_every_steps=50,
    translation_threshold_m=.05, rotation_threshold_deg=15., max_initial_penetration_m=.01,
    contact_margin_m=.01, contact_threshold_m=.002,
    wrist='fixed at decoded pose', joints='native Sharpa XML position actuators; no artificial squeeze',
    stages='single predicted grasp pose; no generated pregrasp or squeeze',
    object_collision='CoACD decomposition of the archived full mesh; never whole-object convex hull',
    hand_collision='official Sharpa XML collision geoms and exclusions, native solref/solimp',
    mass_distribution='convex-piece volume-weighted uniform density, normalized to 0.1 kg',
    setting='floating, no table; source tabletop poses are evaluated as floating grasps',
    success='all six independently reset force tests pass; IK fit is reported separately',
    reference='https://arxiv.org/html/2606.09314v1#S5.SS1',
    comparability='adapted Sharpa diagnostic, not paper-comparable GSR/OSR')


def numbers(a):
    return ' '.join(format(float(x), '.17g') for x in np.asarray(a).ravel())


def quaternion(rotation):
    return Rotation.from_matrix(rotation).as_quat(scalar_first=True)


def contact_metrics(contacts, *, margin=.01, threshold=.002):
    """Contacts are (body1, body2, signed surface distance in metres).

    Positive distances are separation; absent finger contacts are capped at the
    query margin, as in DexGraspBench. CDC keeps negative penetration distances.
    """
    finger_dist = {f: margin for f in FINGERS}
    links, ho, hh = set(), [], []
    for a, b, distance in contacts:
        if a == 'object' or b == 'object':
            hand = b if a == 'object' else a
            ho.append(distance)
            for finger in FINGERS:
                if hand.startswith('right_'+finger+'_'):
                    finger_dist[finger] = min(finger_dist[finger], distance)
                    if abs(distance) < threshold:
                        links.add(hand)
                    break
        else:
            hh.append(distance)
    distances = list(finger_dist.values())
    return dict(hand_object_penetration_mm=1000*max(0., -min(ho, default=0.)),
                self_penetration_mm=1000*max(0., -min(hh, default=0.)),
                contact_distance_mm=1000*float(np.maximum(distances, 0).mean()),
                contact_distance_consistency_mm=1000*(max(distances)-min(distances)),
                contact_link_count=len(links),
                finger_distances_mm={f:1000*d for f,d in finger_dist.items()})


def pose_error(position, quat, initial_position, initial_quat):
    translation = float(np.linalg.norm(np.asarray(position)-initial_position))
    dot = float(abs(np.dot(quat, initial_quat))/max(np.linalg.norm(quat)*np.linalg.norm(initial_quat), 1e-30))
    angle = float(np.rad2deg(2*np.arccos(np.clip(dot, 0., 1.))))
    return translation, angle


def pca_diversity(features):
    """Unstandardized [object-relative t(m), official wrist rotvec(rad), q(rad)]."""
    x = np.asarray(features, dtype=float)
    if len(x) < 2:
        return None
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('finite pose features required')
    singular = np.linalg.svd(x-x.mean(0), compute_uv=False)
    variance = singular**2
    if variance.sum() <= 1e-24:
        return None
    return (100*np.cumsum(variance[:5])/variance.sum()).tolist()


class PoseEvaluator:
    def __init__(self, hand, hand_xml, pieces, scale):
        self.hand = hand
        root = ET.parse(hand_xml).getroot()
        root.find('compiler').set('meshdir', str(Path(hand_xml).parent/'meshes'))
        root.find('compiler').set('balanceinertia', 'true')
        # Collapsed checkpoints can create many simultaneous self contacts.
        # Reserve solver workspace explicitly; this does not change dynamics.
        size = root.find('size')
        if size is None:
            size = ET.SubElement(root, 'size')
        size.set('memory', '128M')
        ET.SubElement(root, 'option', timestep='.004', gravity='0 0 0',
                               integrator='implicitfast', cone='elliptic', noslip_iterations='2', impratio='10')
        world = root.find('worldbody')
        wrapper = ET.Element('body', name='eval_wrist', mocap='true')
        for body in list(world):
            world.remove(body)
            wrapper.append(body)
        world.append(wrapper)
        # Kinematic mocap parent holds the wrist; finger joints remain dynamic.
        obj = ET.SubElement(world, 'body', name='object')
        ET.SubElement(obj, 'freejoint', name='object_free')
        assets = root.find('asset')
        for i, piece in enumerate(pieces):
            ET.SubElement(assets, 'mesh', name=f'object_piece_{i}', file=str(Path(piece).resolve()), scale=numbers(scale))
            ET.SubElement(obj, 'geom', type='mesh', mesh=f'object_piece_{i}', name=f'object_geom_{i}')
        for geom in root.iter('geom'):
            geom.set('friction', '.6 .02 .001')
            geom.set('condim', '4')
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        self.object_id = self.model.body('object').id
        ratio = .1/self.model.body_mass[self.object_id]
        self.model.body_mass[self.object_id] *= ratio
        self.model.body_inertia[self.object_id] *= ratio
        mujoco.mj_setConst(self.model, self.data)
        self.object_qadr = self.model.jnt_qposadr[self.model.joint('object_free').id]
        self.qadr = np.array([self.model.jnt_qposadr[self.model.joint(n).id] for n in hand.spec['joint_order']])
        self.ctrl = np.array([self.model.actuator(n+'_ctrl').id for n in hand.spec['joint_order']])
        self.object_geoms = np.flatnonzero(self.model.geom_bodyid == self.object_id)
        self.names = [self.model.body(i).name for i in range(self.model.nbody)]
        self.official_from_grasp = np.asarray(hand.spec['official_from_grasp_rotation'])

    def reset(self, rotation, translation, q, *, forward=True):
        mujoco.mj_resetData(self.model, self.data)
        self.data.mocap_pos[0] = translation
        self.data.mocap_quat[0] = quaternion(np.asarray(rotation) @ self.official_from_grasp.T)
        self.data.qpos[self.qadr] = q
        self.data.ctrl[self.ctrl] = q
        self.data.qpos[self.object_qadr:self.object_qadr+7] = [0, 0, 0, 1, 0, 0, 0]
        if forward:
            mujoco.mj_forward(self.model, self.data)
        else:
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)

    def fk_error(self, rotation, translation, q):
        self.reset(rotation, translation, q, forward=False)
        frames = {}
        for name in self.names:
            if name:
                body = self.data.body(name)
                frame = np.eye(4)
                frame[:3, :3], frame[:3, 3] = body.xmat.reshape(3, 3), body.xpos
                frames[name] = frame
        # Official XML omits URDF fingertip marker bodies. Restore only fixed
        # marker transforms, never moving joints, from the recorded hand spec.
        for parent, child, origin, _, kind, _ in self.hand.edges:
            if child not in frames and parent in frames and kind == 'fixed':
                frames[child] = frames[parent] @ origin
        actual = []
        for kp in self.hand.spec['keypoints']:
            frame = frames[kp['link']]
            actual.append(frame[:3, 3]+frame[:3, :3] @ kp['local_xyz_m'])
        expected = self.hand.world_keypoints(rotation, translation, q)
        return float(np.linalg.norm(np.asarray(actual)-expected, axis=1).max())

    def geometry(self):
        self.model.geom_margin[self.object_geoms] = PROTOCOL['contact_margin_m']
        # Distance queries need collision detection, not a constraint solve.
        # Severe overlapping poses are screened before dynamic simulation.
        mujoco.mj_collision(self.model, self.data)
        contacts = [(self.names[self.model.geom_bodyid[c.geom1]],
                     self.names[self.model.geom_bodyid[c.geom2]], float(c.dist)) for c in self.data.contact]
        result = contact_metrics(contacts)
        self.model.geom_margin[self.object_geoms] = 0
        return result

    def evaluate(self, rotation, translation, q, *, simulate=True):
        rotation, translation, q = map(np.asarray, (rotation, translation, q))
        if (rotation.shape != (3, 3) or translation.shape != (3,) or q.shape != (22,)
                or not all(np.isfinite(x).all() for x in (rotation, translation, q))
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)
                or not self.hand.in_limits(q)):
            raise ValueError('invalid finite hand pose or joint limits')
        self.reset(rotation, translation, q, forward=False)
        result = self.geometry()
        result.update(geometry_evaluated=True, simulation_evaluated=False, single_pose_force_success=None)
        if not simulate:
            return result
        result.update(simulation_evaluated=True, single_pose_force_success=False, force_trials=[])
        if max(result['hand_object_penetration_mm'], result['self_penetration_mm']) > 10:
            result['failure_reason'] = 'initial_penetration_over_10mm'
            return result
        for direction in PROTOCOL['force_directions']:
            self.reset(rotation, translation, q)
            initial = self.data.qpos[self.object_qadr:self.object_qadr+7].copy()
            max_pos, max_angle = 0., 0.
            passed = True
            steps_done = 0
            for step in range(0, PROTOCOL['settling_steps']+PROTOCOL['force_steps'], 50):
                if step >= PROTOCOL['settling_steps']:
                    self.data.xfrc_applied[self.object_id, :3] = np.asarray(direction)  # 10 m/s² * .1 kg
                mujoco.mj_step(self.model, self.data, nstep=50)
                steps_done += 50
                state = self.data.qpos[self.object_qadr:self.object_qadr+7]
                p, a = pose_error(state[:3], state[3:], initial[:3], initial[3:])
                max_pos, max_angle = max(max_pos, p), max(max_angle, a)
                if (not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
                        or np.any(self.data.warning.number) or self.data.time < steps_done*.004-1e-6):
                    passed = False
                    result['failure_reason'] = 'simulation_warning_or_nonfinite'
                    break
                if p >= .05 or a >= 15:
                    passed = False
                    result['failure_reason'] = 'object_displacement'
                    break
            result['force_trials'].append(dict(direction=direction, passed=passed, steps=steps_done,
                                              max_translation_m=max_pos, max_rotation_deg=max_angle))
            if not passed:
                break
        result['single_pose_force_success'] = (len(result['force_trials']) == 6 and
                                              all(t['passed'] for t in result['force_trials']))
        return result
