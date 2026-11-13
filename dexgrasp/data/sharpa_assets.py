"""Sharpa diagnostic hand mapping and box mesh helpers.

Project-defined landmarks and palm patch; not hardware calibration.
"""
import json
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from dexgrasp.geometry.hand import URDFHand
from dexgrasp.geometry.mesh import validate_closed_mesh


def box(extents, center=(0, 0, 0)):
    vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1)
                         for z in (-1, 1)], float) * np.asarray(extents)/2 + center
    hull = ConvexHull(vertices)
    faces = hull.simplices.copy()
    for i, face in enumerate(faces):
        a, b, c = vertices[face]
        if np.dot(np.cross(b-a, c-a), hull.equations[i, :3]) < 0:
            faces[i] = face[::-1]
    validate_closed_mesh(vertices, faces)
    return vertices, faces


def make_hand(source, destination):
    destination.mkdir()
    for name in ('right_sharpa_wave.urdf', 'LICENSE.txt', 'NOTICE.txt', 'source.json'):
        shutil.copyfile(source/name, destination/name)
    tree = ET.parse(source/'right_sharpa_wave.urdf')
    robot = tree.getroot()
    joints = {j.attrib['name']: j for j in robot.findall('joint')}
    order = [name for name, j in joints.items() if j.attrib['type'] == 'revolute']
    # New root axes in the official root: x=+y, y=+z, z=+x (palm-facing).
    official_from_grasp = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
    ET.SubElement(robot, 'link', name='diagnostic_grasp_frame')
    joint = ET.SubElement(robot, 'joint', name='diagnostic_frame_fixed', type='fixed')
    ET.SubElement(joint, 'parent', link='diagnostic_grasp_frame')
    ET.SubElement(joint, 'child', link='right_hand_C_MC')
    rpy = Rotation.from_matrix(official_from_grasp.T).as_euler('xyz')
    ET.SubElement(joint, 'origin', xyz='0 0 0', rpy=' '.join(map(str, rpy)))
    tree.write(destination/'diagnostic_hand.urdf', encoding='utf-8', xml_declaration=True)
    points = [dict(name='wrist', link='right_hand_C_MC', local_xyz_m=[0., 0., 0.], palm_anchor=True)]
    for finger in ('thumb', 'index', 'middle', 'ring', 'pinky'):
        landmarks = ('CMC_FE', 'MCP_FE', 'IP') if finger == 'thumb' else ('MCP_FE', 'PIP', 'DIP')
        for landmark in landmarks:
            j = joints[f'right_{finger}_{landmark}']
            parent = j.find('parent').attrib['link']
            points.append(dict(name=f'{finger}_{landmark}', link=parent,
                               local_xyz_m=list(map(float, j.find('origin').attrib['xyz'].split())),
                               palm_anchor=parent == 'right_hand_C_MC'))
        points.append(dict(name=f'{finger}_tip', link=f'right_{finger}_fingertip',
                           local_xyz_m=[0., 0., 0.], palm_anchor=False))
    # Explicit experimental plane, not a measured hardware skin calibration.
    rays = [[y, z, .012] for y in np.linspace(-.012, .012, 8)
            for z in np.linspace(.04, .064, 8)]
    spec = dict(schema_version=1, hand_model='sharpa', calibrated=True,
                calibration_scope='numerically defined kinematic diagnostic only; not hardware/KPGrasp calibration',
                root_link='diagnostic_grasp_frame', urdf='diagnostic_hand.urdf',
                joint_order=order, keypoints=points, approach_axis=[0, 0, 1],
                reference_length_m=.2, keypoint_map_version='project-urdf-landmarks-diagnostic-v1',
                label_version='kinematic-wrist-z-v2',
                direction_condition=dict(kind='official_wrist_positive_z_v2', local_axis=[0, 1, 0],
                                         official_axis='+Zh', sign='wrist toward fingers'),
                palm=dict(normal=[0, 0, 1], ray_origins_m=rays),
                official_from_grasp_rotation=official_from_grasp.tolist())
    path = destination/'hand_spec.json'
    path.write_text(json.dumps(spec, indent=2)+'\n')
    hand = URDFHand(path)
    # Reference length is the zero-pose wrist-to-middle-tip distance in this URDF.
    spec['reference_length_m'] = float(np.linalg.norm(hand.fk(np.zeros(22))[12]))
    path.write_text(json.dumps(spec, indent=2)+'\n')
    return URDFHand(path)
