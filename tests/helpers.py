"""Synthetic software fixtures only. These are not Sharpa calibration or training data."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from dexgrasp.config import ModelConfig, DEFAULT_RUN
from dexgrasp.data.dataset import sha256
from dexgrasp.data.prepare import ShardWriter
from dexgrasp.geometry.hand import URDFHand
from dexgrasp.geometry.mesh import sample_observation
from dexgrasp.models.encoders import farthest_point_indices

torch.set_num_threads(1)


class FixtureEncoder(nn.Module):
    """Test double confined to tests; production has only MinkUNet."""
    def __init__(self, cfg):
        super().__init__()
        self.project = nn.Linear(4, cfg.width)
        self.tokens = cfg.tokens
        self.calls = 0

    def forward(self, points, mask):
        self.calls += 1
        indices = farthest_point_indices(points, self.tokens)
        rows = torch.arange(len(points), device=points.device)[:, None]
        centers = points[rows, indices]
        features = self.project(torch.cat((points, mask[..., None].float()), -1))
        return features[rows, indices], centers


def small_config():
    return ModelConfig(width=32, depth=2, heads=4, tokens=8, backbone_channels=16,
                       hand_reference_length_m=0.1)


def condition(batch=2, device='cpu'):
    return dict(object_points=torch.randn(batch, 1024, 3, device=device),
                scene_points=torch.randn(batch, 2048, 3, device=device),
                scene_mask=torch.cat((torch.ones(batch, 512), torch.zeros(batch, 1536)), 1).to(device),
                palm_gap=torch.full((batch, 1), .02, device=device),
                grasp_direction_cmd=torch.tensor([[0., 0., 1.]], device=device).repeat(batch, 1))


def box_mesh(z=0.02):
    v = np.array([[-.1, -.1, z], [.1, -.1, z], [.1, .1, z], [-.1, .1, z],
                  [-.1, -.1, z+.15], [.1, -.1, z+.15], [.1, .1, z+.15], [-.1, .1, z+.15]])
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int64)
    return v, f


def hand_fixture(root):
    folder = Path(root)/'hand'
    folder.mkdir(parents=True, exist_ok=True)
    xml = ['<robot name="synthetic_fixture"><link name="palm"/>']
    for i in range(22):
        parent = 'palm' if i == 0 else f'link{i-1}'
        xml.append(f'<link name="link{i}"/><joint name="j{i}" type="revolute">'
                   f'<parent link="{parent}"/><child link="link{i}"/>'
                   '<origin xyz="0.005 0 0"/><axis xyz="0 1 0"/>'
                   '<limit lower="-0.5" upper="0.5" effort="1" velocity="1"/></joint>')
    xml.append('</robot>')
    (folder/'fixture.urdf').write_text(''.join(xml))
    points = [dict(name=f'p{i}', link='palm', local_xyz_m=pos, palm_anchor=True)
              for i, pos in enumerate(([0, 0, 0], [.03, 0, 0], [0, .03, 0]))]
    points += [dict(name=f'p{i+3}', link=f'link{link}', local_xyz_m=[.004, 0, .002], palm_anchor=False)
               for i, link in enumerate([*range(17), 21])]
    rays = [[float(x), float(y), 0.] for x in np.linspace(-.02, .02, 8) for y in np.linspace(-.02, .02, 8)]
    spec = dict(schema_version=1, hand_model='sharpa', calibrated=True, fixture_only=True,
                root_link='palm', urdf='fixture.urdf', joint_order=[f'j{i}' for i in range(22)],
                reference_length_m=.1, keypoints=points, approach_axis=[0, 0, 1],
                palm=dict(normal=[0, 0, 1], ray_origins_m=rays),
                keypoint_map_version='synthetic-test-only', label_version='synthetic-test-only')
    path = folder/'hand_spec.json'
    path.write_text(json.dumps(spec))
    return path


def dataset_fixture(root, *, raw=False):
    root = Path(root)
    spec_path = hand_fixture(root)
    hand = URDFHand(spec_path)
    (root/'scenes').mkdir()
    (root/'assets').mkdir()
    v, f = box_mesh()
    np.savez(root/'assets/target.npz', vertices=v, faces=f)
    np.savez(root/'assets/table.npz', vertices=v-np.array([0, 0, .3]), faces=f)
    rng = np.random.default_rng(0)
    if not raw:
        (root/'shards').mkdir()
        (root/'splits').mkdir()
        writer = ShardWriter(root/'shards/part-00000.h5', spec_path)
    for i, split in enumerate(('train', 'val', 'test')):
        scene = dict(scene_id=f's{i}', view_id='v0', object_id=f'object{i}', split=split,
                     target_id='target', camera=None, collision_validation_version='synthetic-not-physical',
                     instances=[dict(instance_id=key, mesh=f'assets/{key}.npz', scale=1.,
                                     R=np.eye(3).tolist(), t=[0, 0, 0]) for key in ('target', 'table')],
                     grasps=f'assets/grasps{i}.npz')
        (root/'scenes'/f's{i}.json').write_text(json.dumps(scene))
        np.savez(root/f'assets/grasps{i}.npz', R=np.tile(np.eye(3), (2, 1, 1)).astype(np.float32),
                 t=np.zeros((2, 3), np.float32), q=np.zeros((2, 22), np.float32),
                 valid=np.ones(2, bool), grasp_id=np.array(['g0', 'g1']))
        if not raw:
            obs = sample_observation((v, f), (v-np.array([0, 0, .3]), f), rng)
            obs.update(scene_id=f's{i}', view_id='v0', object_id=f'object{i}')
            oi = writer.observation(obs)
            lines = []
            for j in range(2):
                row = dict(observation_index=oi, grasp_id=f'g{j}', R=np.eye(3, dtype=np.float32),
                           t=np.zeros(3, np.float32), q=np.zeros(22, np.float32),
                           K=hand.fk(np.zeros(22)).astype(np.float32), d=np.array([0, 0, 1], np.float32),
                           h=np.array([.02], np.float32), hit_ratio=np.float32(1))
                gi = writer.grasp(row)
                lines.append(json.dumps(dict(shard='shards/part-00000.h5', grasp_index=gi)))
            (root/'splits'/f'{split}.jsonl').write_text('\n'.join(lines)+'\n')
    if not raw:
        writer.close()
    return hand


def run_config(root, name='fm', teacher=None):
    root = Path(root)
    cfg = {**DEFAULT_RUN, 'method': name, 'device': 'cpu', 'num_workers': 0,
           'batch_size': 2, 'accumulation': 2, 'max_steps': 4, 'save_every': 2, 'val_every': 2,
           'val_batches': 1, 'sample_steps': 1, 'teacher_steps': 2,
           'dataset_root': str(root/'dataset'), 'hand_spec': str(root/'dataset/hand/hand_spec.json'),
           'run_dir': str(root/f'{name}_run'), 'checkpoint_dir': str(root/f'{name}_ckpt'),
           'model': dict(width=32, depth=1, heads=4, tokens=8, backbone_channels=16)}
    if teacher:
        cfg['teacher_checkpoint'] = str(teacher)
    return cfg
