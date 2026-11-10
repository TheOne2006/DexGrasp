"""Data v1 HDF5: observation sharing, strict labels and worker-local bounded caches."""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

SPLITS = ('train', 'val', 'test')
FLOAT_SHAPES = {'object_points': (1024, 3), 'scene_points': (2048, 3),
                'R': (3, 3), 't': (3,), 'q': (22,), 'K': (21, 3),
                'd': (3,), 'h': (1,), 'hit_ratio': ()}
OBS_KEYS = ('object_points', 'scene_points', 'scene_mask', 'scene_id', 'view_id', 'object_id')
GRASP_KEYS = ('observation_index', 'grasp_id', 'R', 't', 'q', 'K', 'd', 'h', 'hit_ratio')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def contained(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('dataset paths must stay inside dataset_root')
    return path


def read_split(root, split):
    if split not in SPLITS:
        raise ValueError('split must be train, val or test')
    path = Path(root) / 'splits' / f'{split}.jsonl'
    rows = []
    seen = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) != {'shard', 'grasp_index'} or type(row['grasp_index']) is not int or row['grasp_index'] < 0:
            raise ValueError(f'invalid split row in {path}')
        key = (str(contained(root, row['shard'])), row['grasp_index'])
        if key in seen:
            raise ValueError('duplicate grasp in split')
        seen.add(key)
        rows.append(row)
    if not rows:
        raise ValueError(f'empty {split} split')
    return rows


def validate_observation(obs):
    for key in ('object_points', 'scene_points'):
        x = np.asarray(obs[key])
        if x.shape != FLOAT_SHAPES[key] or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError(f'invalid {key}: expected finite float32 {FLOAT_SHAPES[key]}')
    mask = np.asarray(obs['scene_mask'])
    if mask.shape != (2048,) or mask.dtype != np.uint8 or not np.isin(mask, [0, 1]).all() or mask.sum() != 512:
        raise ValueError('scene_mask must be uint8 with 512 target points')
    for key in ('scene_id', 'view_id', 'object_id'):
        if not text(obs[key]):
            raise ValueError(f'missing {key}')


def validate_grasp(grasp, ray_count):
    for key in ('R', 't', 'q', 'K', 'd', 'h', 'hit_ratio'):
        x = np.asarray(grasp[key])
        if x.shape != FLOAT_SHAPES[key] or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError(f'invalid {key}: expected finite float32 {FLOAT_SHAPES[key]}')
    rotation = grasp['R']
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=2e-5):
        raise ValueError('R must be a proper rotation')
    if not np.allclose(grasp['d'], rotation[:, 2], atol=2e-5):
        raise ValueError('d must equal the third column of R')
    if grasp['h'][0] < 0 or not 0 <= grasp['hit_ratio'] <= 1:
        raise ValueError('invalid gap or hit ratio')
    if ray_count < 64 or float(grasp['hit_ratio']) * ray_count < 64 - 1e-4:
        raise ValueError('palm_gap needs at least 64 hits')
    if not text(grasp['grasp_id']):
        raise ValueError('missing grasp_id')


class GraspDataset(Dataset):
    def __init__(self, root, split, coord_scale_m, *, observation_cache_size=64, max_open_shards=8):
        self.root, self.split = Path(root).resolve(), split
        if not np.isfinite(coord_scale_m) or coord_scale_m <= 0:
            raise ValueError('coordinate scale must be positive')
        if observation_cache_size < 0 or max_open_shards < 1:
            raise ValueError('invalid cache size')
        self.scale = float(coord_scale_m)
        self.rows = read_split(root, split)
        spec_path = self.root / 'hand/hand_spec.json'
        self.spec_hash = sha256(spec_path)
        spec = json.loads(spec_path.read_text())
        self.ray_count = len(spec['palm']['ray_origins_m'])
        self.cache_size, self.max_open = observation_cache_size, max_open_shards
        self._pid, self._handles, self._observations = os.getpid(), OrderedDict(), OrderedDict()

    def __len__(self):
        return len(self.rows)

    def close(self):
        for f in self._handles.values():
            f.close()
        self._handles.clear()
        self._observations.clear()

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_handles'], state['_observations'] = OrderedDict(), OrderedDict()
        return state

    def _file(self, relative):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        path = str(contained(self.root, relative))
        if path not in self._handles:
            f = h5py.File(path, 'r')
            if (f.attrs.get('schema_version') != 1 or f.attrs.get('hand_spec_sha256') != self.spec_hash
                    or f.attrs.get('point_sampling') != 'full_mesh_area_uniform'):
                f.close()
                raise ValueError('shard schema, hand identity or sampling protocol mismatch')
            self._handles[path] = f
            if len(self._handles) > self.max_open:
                self._handles.popitem(last=False)[1].close()
        self._handles.move_to_end(path)
        return self._handles[path]

    def raw(self, index):
        row = self.rows[index]
        f = self._file(row['shard'])
        gi = row['grasp_index']
        if gi >= len(f['grasps/observation_index']):
            raise ValueError('grasp index out of bounds')
        grasp = {k: f[f'grasps/{k}'][gi] for k in GRASP_KEYS}
        oi = int(grasp['observation_index'])
        if oi < 0 or oi >= len(f['observations/object_points']):
            raise ValueError('observation index out of bounds')
        key = (row['shard'], oi)
        if key not in self._observations:
            obs = {k: f[f'observations/{k}'][oi] for k in OBS_KEYS}
            validate_observation(obs)
            if self.cache_size:
                self._observations[key] = obs
                if len(self._observations) > self.cache_size:
                    self._observations.popitem(last=False)
        else:
            self._observations.move_to_end(key)
            obs = self._observations[key]
        validate_grasp(grasp, self.ray_count)
        return obs, grasp

    def __getitem__(self, index):
        obs, grasp = self.raw(index)
        center = obs['object_points'].mean(0, dtype=np.float64).astype(np.float32)
        condition = {k: torch.from_numpy((obs[k] - center) / self.scale)
                     for k in ('object_points', 'scene_points')}
        condition.update(scene_mask=torch.from_numpy(obs['scene_mask'].astype(bool)),
                         palm_gap=torch.from_numpy(grasp['h'].copy()),
                         grasp_direction_cmd=torch.from_numpy(grasp['d'].copy()))
        return dict(condition=condition,
                    keypoints_gt=torch.from_numpy((grasp['K'] - center) / self.scale),
                    metadata=dict(center_m=torch.from_numpy(center), coord_scale_m=self.scale,
                                  scene_id=text(obs['scene_id']), view_id=text(obs['view_id']),
                                  object_id=text(obs['object_id']), grasp_id=text(grasp['grasp_id']),
                                  shard=self.rows[index]['shard'], grasp_index=self.rows[index]['grasp_index'],
                                  split=self.split))


def audit_dataset(root, coord_scale_m=0.2, *, deep=True):
    """Full split leakage audit. Deep mode validates every referenced grasp, not a sample."""
    root = Path(root).resolve()
    hand = None
    if deep:
        from ..geometry.hand import URDFHand
        hand = URDFHand(root / 'hand/hand_spec.json')
    objects, scenes, references, grasp_ids = {}, {}, {}, {}
    counts, files = {}, {}
    for split in SPLITS:
        ds = GraspDataset(root, split, coord_scale_m, observation_cache_size=8)
        try:
            counts[split] = len(ds)
            for i, row in enumerate(ds.rows):
                f = ds._file(row['shard'])
                for group, keys in (('observations', OBS_KEYS), ('grasps', GRASP_KEYS)):
                    sizes = {len(f[f'{group}/{k}']) for k in keys}
                    if len(sizes) != 1:
                        raise ValueError(f'{group} dataset lengths disagree')
                if f['grasps/observation_index'].dtype != np.int64:
                    raise ValueError('observation_index must be int64')
                gi = row['grasp_index']
                oi = int(f['grasps/observation_index'][gi])
                if not 0 <= oi < len(f['observations/scene_id']):
                    raise ValueError('observation index out of bounds')
                oid = text(f['observations/object_id'][oi])
                sid = text(f['observations/scene_id'][oi])
                gid = (sid, text(f['observations/view_id'][oi]), text(f['grasps/grasp_id'][gi]))
                for mapping, key in ((objects, oid), (scenes, sid),
                                     (references, (row['shard'], gi))):
                    if key in mapping and mapping[key] != split:
                        raise ValueError(f'cross-split leakage: {key}')
                    mapping[key] = split
                if gid in grasp_ids:
                    raise ValueError(f'duplicate physical grasp identity: {gid}')
                grasp_ids[gid] = split
                if deep:
                    _, grasp = ds.raw(i)
                    q = grasp['q']
                    if not hand.in_limits(q):
                        raise ValueError('joint angles outside calibrated hand limits')
                    rebuilt = hand.world_keypoints(grasp['R'], grasp['t'], q)
                    if not np.allclose(rebuilt, grasp['K'], atol=2e-6, rtol=1e-5):
                        raise ValueError('stored K disagrees with calibrated FK(R,t,q)')
                path = contained(root, row['shard'])
                if row['shard'] not in files:
                    stat = path.stat()
                    files[row['shard']] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
                    if deep:
                        files[row['shard']]['sha256'] = sha256(path)
        finally:
            ds.close()
    return dict(schema_version=1, samples=counts, objects=len(objects), scenes=len(scenes),
                hand_spec_sha256=sha256(root / 'hand/hand_spec.json'),
                split_sha256={s: sha256(root / 'splits' / f'{s}.jsonl') for s in SPLITS},
                shard_stats=files, sampling='full_mesh_area_uniform', deep=deep)
