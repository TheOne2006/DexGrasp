"""Explicit Sharpa calibration and URDF FK/Jacobians; no invented link mapping."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def vector(value, size=3):
    v = np.asarray(value, dtype=np.float64)
    if v.shape != (size,) or not np.isfinite(v).all():
        raise ValueError(f'expected a finite {size}-vector')
    return v


def rotation_axis(axis, angle):
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*cross + (1-np.cos(angle))*(cross @ cross)


def rotation_rpy(rpy):
    r, p, y = rpy
    return rotation_axis([0, 0, 1], y) @ rotation_axis([0, 1, 0], p) @ rotation_axis([1, 0, 0], r)


def read_hand_spec(path, *, require_calibrated=True):
    spec = json.loads(Path(path).read_text())
    if spec.get('schema_version') != 1 or spec.get('hand_model') != 'sharpa':
        raise ValueError('expected Sharpa hand spec version 1')
    if require_calibrated and spec.get('calibrated') is not True:
        raise ValueError('Sharpa hand_spec is not numerically calibrated')
    if not require_calibrated and spec.get('calibrated') is not True:
        return spec
    length = spec.get('reference_length_m')
    if not isinstance(length, (float, int)) or not np.isfinite(length) or length <= 0:
        raise ValueError('calibrated reference_length_m required')
    if not np.array_equal(vector(spec['approach_axis']), [0., 0., 1.]):
        raise ValueError('Data v1 uses local +z as approach axis')
    joints, points = spec['joint_order'], spec['keypoints']
    if len(joints) != 22 or len(set(joints)) != 22 or not all(isinstance(j, str) and j for j in joints):
        raise ValueError('22 unique ordered joint names required')
    if len(points) != 21 or len({p['name'] for p in points}) != 21:
        raise ValueError('21 unique ordered keypoint mappings required')
    for p in points:
        vector(p['local_xyz_m'])
        if not p['link'] or type(p['palm_anchor']) is not bool:
            raise ValueError('each keypoint needs a link and explicit palm_anchor flag')
    if sum(p['palm_anchor'] for p in points) < 3:
        raise ValueError('at least three rigid palm anchors required')
    normal = vector(spec['palm']['normal'])
    origins = np.asarray(spec['palm']['ray_origins_m'], dtype=np.float64)
    if origins.ndim != 2 or origins.shape[1] != 3 or len(origins) < 64 or not np.isfinite(origins).all():
        raise ValueError('at least 64 calibrated palm ray origins required')
    if not np.isclose(np.linalg.norm(normal), 1, atol=1e-6):
        raise ValueError('palm normal must be unit length')
    if np.ptp(origins @ normal) > 1e-6 or np.linalg.matrix_rank(origins-origins.mean(0), tol=1e-8) < 2:
        raise ValueError('palm origins must span a planar measurement region')
    for key in ('root_link', 'urdf', 'keypoint_map_version', 'label_version'):
        if not isinstance(spec.get(key), str) or not spec[key]:
            raise ValueError(f'missing hand spec {key}')
    return spec


class URDFHand:
    def __init__(self, spec_path):
        self.spec_path = Path(spec_path).resolve()
        self.spec = read_hand_spec(self.spec_path)
        self.urdf_path = (self.spec_path.parent / self.spec['urdf']).resolve()
        root = ET.parse(self.urdf_path).getroot()
        root_link = self.spec['root_link']
        links = {e.attrib['name'] for e in root.findall('link')}
        if root_link not in links:
            raise ValueError('root_link missing from URDF')
        children, child_links, names = {}, set(), set()
        for e in root.findall('joint'):
            name, kind = e.attrib['name'], e.attrib['type']
            parent, child = e.find('parent').attrib['link'], e.find('child').attrib['link']
            if parent not in links or child not in links or child in child_links or name in names:
                raise ValueError('invalid URDF tree')
            child_links.add(child)
            names.add(name)
            children.setdefault(parent, []).append((name, kind, child, e))
        self.edges, self.ancestors = [], {root_link: []}
        queue = [root_link]
        order = {name: i for i, name in enumerate(self.spec['joint_order'])}
        self.lower, self.upper = np.empty(22), np.empty(22)
        seen_moving = set()
        while queue:
            parent = queue.pop(0)
            for name, kind, child, e in children.get(parent, []):
                if child in self.ancestors:
                    raise ValueError('cycle in hand URDF')
                if kind not in ('fixed', 'revolute', 'prismatic') or e.find('mimic') is not None:
                    raise ValueError('v1 URDF adapter needs explicit fixed/revolute/prismatic joints without mimic')
                origin = e.find('origin')
                xyz = vector([float(v) for v in (origin.get('xyz', '0 0 0') if origin is not None else '0 0 0').split()])
                rpy = vector([float(v) for v in (origin.get('rpy', '0 0 0') if origin is not None else '0 0 0').split()])
                transform = np.eye(4)
                transform[:3, :3], transform[:3, 3] = rotation_rpy(rpy), xyz
                axis_e = e.find('axis')
                axis = vector([float(v) for v in (axis_e.get('xyz', '1 0 0') if axis_e is not None else '1 0 0').split()])
                index = None
                if kind != 'fixed':
                    if name not in order or np.linalg.norm(axis) < 1e-8:
                        raise ValueError('moving joint missing from joint_order or zero axis')
                    axis /= np.linalg.norm(axis)
                    index = order[name]
                    limit = e.find('limit')
                    if limit is None:
                        raise ValueError('bounded joints required for IK')
                    self.lower[index], self.upper[index] = float(limit.get('lower')), float(limit.get('upper'))
                    seen_moving.add(name)
                self.edges.append((parent, child, transform, axis, kind, index))
                self.ancestors[child] = self.ancestors[parent] + ([] if index is None else [index])
                queue.append(child)
        if seen_moving != set(order) or not np.isfinite(self.lower).all() or not np.isfinite(self.upper).all() or (self.lower >= self.upper).any():
            raise ValueError('URDF must expose exactly 22 finite bounded joints under root_link')
        self.palm_indices = np.array([i for i, p in enumerate(self.spec['keypoints']) if p['palm_anchor']])
        for p in self.spec['keypoints']:
            if p['link'] not in self.ancestors:
                raise ValueError('keypoint link is not below root_link')
            if p['palm_anchor'] and self.ancestors[p['link']]:
                raise ValueError('palm anchors must be rigid with respect to root_link')
        self.palm_template = self.fk((self.lower+self.upper)/2)[self.palm_indices]
        if np.linalg.matrix_rank(self.palm_template-self.palm_template.mean(0), tol=1e-8) < 2:
            raise ValueError('palm anchors must be non-collinear')

    def fk(self, q, *, jacobian=False):
        q = vector(q, 22)
        transforms = {self.spec['root_link']: np.eye(4)}
        joint_origins, joint_axes = np.zeros((22, 3)), np.zeros((22, 3))
        kinds = [None]*22
        for parent, child, origin, axis, kind, index in self.edges:
            frame = transforms[parent] @ origin
            motion = np.eye(4)
            if index is not None:
                joint_origins[index], joint_axes[index], kinds[index] = frame[:3, 3], frame[:3, :3] @ axis, kind
                if kind == 'revolute':
                    motion[:3, :3] = rotation_axis(axis, q[index])
                else:
                    motion[:3, 3] = axis*q[index]
            transforms[child] = frame @ motion
        points, jac = [], np.zeros((21, 3, 22))
        for j, kp in enumerate(self.spec['keypoints']):
            frame = transforms[kp['link']]
            p = frame[:3, :3] @ kp['local_xyz_m'] + frame[:3, 3]
            points.append(p)
            if jacobian:
                for i in self.ancestors[kp['link']]:
                    jac[j, :, i] = (np.cross(joint_axes[i], p-joint_origins[i])
                                    if kinds[i] == 'revolute' else joint_axes[i])
        points = np.asarray(points)
        return (points, jac) if jacobian else points

    def world_keypoints(self, rotation, translation, q):
        return self.fk(q) @ np.asarray(rotation).T + translation

    def in_limits(self, q, tolerance=1e-6):
        """Allow float32 storage roundoff at URDF limits; never clip data labels."""
        q = vector(q, 22)
        return bool((q >= self.lower-tolerance).all() and (q <= self.upper+tolerance).all())
