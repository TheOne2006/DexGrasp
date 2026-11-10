"""Build a kinematic-only diagnostic dataset, NOT physics-validated grasp data.

Official Sharpa joint geometry; project-defined 21 landmarks and palm patch.
No KPGrasp marker calibration or successful-grasp labels are claimed.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dexgrasp.data.dataset import audit_dataset, sha256
from dexgrasp.data.prepare import ShardWriter
from dexgrasp.geometry.hand import URDFHand
from dexgrasp.geometry.mesh import sample_observation, measure_palm_gap, validate_closed_mesh


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
                label_version='kinematic-only-v1',
                palm=dict(normal=[0, 0, 1], ray_origins_m=rays),
                official_from_grasp_rotation=official_from_grasp.tolist())
    path = destination/'hand_spec.json'
    path.write_text(json.dumps(spec, indent=2)+'\n')
    hand = URDFHand(path)
    # Reference length is the zero-pose wrist-to-middle-tip distance in this URDF.
    spec['reference_length_m'] = float(np.linalg.norm(hand.fk(np.zeros(22))[12]))
    path.write_text(json.dumps(spec, indent=2)+'\n')
    return URDFHand(path)


def build(source, root, seed=20261110):
    root.mkdir(parents=True, exist_ok=False)
    hand = make_hand(source, root/'hand')
    for directory in ('assets', 'scenes', 'shards', 'splits', 'raw_grasps'):
        (root/directory).mkdir()
    rng = np.random.default_rng(seed)
    rows = {s: [] for s in ('train', 'val', 'test')}
    dimensions = [[.060, .065, .070], [.072, .054, .060], [.055, .078, .065],
                  [.080, .060, .055], [.065, .072, .058], [.058, .062, .080]]
    table = box([.40, .40, .015], [0, 0, -.03])
    np.savez(root/'assets/table.npz', vertices=table[0], faces=table[1])
    writer = ShardWriter(root/'shards/part-00000.h5', hand.spec_path)
    gap_records = []
    try:
        for i, dims in enumerate(dimensions):
            split = 'train' if i < 4 else ('val' if i == 4 else 'test')
            object_id, scene_id = f'procedural_cuboid_{i:02d}', f'diagnostic_{i:02d}'
            center = np.array([0., 0., .12])
            target = box(dims, center)
            mesh_path = f'assets/{object_id}.npz'
            np.savez(root/mesh_path, vertices=target[0], faces=target[1])
            obs = sample_observation(target, table, rng)
            obs.update(object_id=object_id, scene_id=scene_id, view_id='full_mesh')
            oi = writer.observation(obs)
            raw = {k: [] for k in ('R', 't', 'q', 'grasp_id')}
            for k in range(2):
                rotation = Rotation.from_euler('zy', [.25*i, 1.2+.12*k]).as_matrix()
                palm_center = np.mean(hand.spec['palm']['ray_origins_m'], axis=0)
                translation = center - rotation@palm_center - rotation[:, 2]*(.060+.018*k)
                q = np.zeros(22)
                for j, name in enumerate(hand.spec['joint_order']):
                    if name.endswith(('FE', 'PIP', 'DIP', '_IP')):
                        q[j] = .25 + .16*k + .025*i
                    elif name.endswith('_CMC'):
                        q[j] = .08
                assert hand.in_limits(q)
                rotation, translation, q = [x.astype(np.float32) for x in (rotation, translation, q)]
                gap = measure_palm_gap(*target, rotation, translation, hand.spec['palm'])
                if not gap['valid']:
                    raise ValueError(f'invalid diagnostic palm rays: {gap}')
                row = dict(observation_index=oi, grasp_id=f'{object_id}_pose{k}', R=rotation,
                           t=translation, q=q, K=hand.world_keypoints(rotation, translation, q).astype(np.float32),
                           d=rotation[:, 2], h=np.array([gap['h']], np.float32),
                           hit_ratio=np.float32(gap['hit_ratio']))
                gi = writer.grasp(row)
                rows[split].append(dict(shard='shards/part-00000.h5', grasp_index=gi))
                for key in raw:
                    raw[key].append(row[key])
                gap_records.append(dict(grasp_id=row['grasp_id'], **gap))
            np.savez(root/f'raw_grasps/{scene_id}.npz', **raw)
            scene = dict(scene_id=scene_id, object_id=object_id, view_id='full_mesh', split=split,
                         target_id=object_id, camera=None, grasps=f'raw_grasps/{scene_id}.npz',
                         collision_validation_version=None, physical_success=None,
                         label_source='procedural joint poses; no contact/collision/physics validation',
                         instances=[dict(instance_id=name, mesh=path, scale=1., R=np.eye(3).tolist(), t=[0, 0, 0])
                                    for name, path in [(object_id, mesh_path), ('table', 'assets/table.npz')]])
            (root/f'scenes/{scene_id}.json').write_text(json.dumps(scene, indent=2)+'\n')
    finally:
        writer.close()
    for split, indices in rows.items():
        (root/f'splits/{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in indices))
    report = dict(seed=seed, scope='synthetic kinematic overfit diagnostic only',
                  source=json.loads((source/'source.json').read_text()),
                  kpgrasp_marker_calibration=False, collision_validated=False, physical_success_rate=None,
                  duplicate_geometry_across_splits=False, palm_ray_checks=gap_records,
                  audit=audit_dataset(root),
                  files_sha256={str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*')) if p.is_file()},
                  builder_sha256=sha256(__file__))
    (root/'preparation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['audit'], indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    build(args.source.resolve(), args.root.resolve())
