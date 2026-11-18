"""Single-frame and paired three-stage HDF5 observations, labels, splits and audit."""
from collections import OrderedDict
from array import array
import hashlib
import json
import os
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from ..models.config import STAGE_ORDER
from ..geometry.direction import direction_definition, official_wrist_definition

SPLITS = ('train', 'test')
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
        raise ValueError('split must be train or test')
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


class TrainingRows:
    """Compact index for audited splits; no millions of dicts copied to workers."""
    def __init__(self, root, split):
        if split not in SPLITS:
            raise ValueError('split must be train or test')
        shards, shard_ids, indices = {}, array('I'), array('q')
        with (Path(root)/'splits'/f'{split}.jsonl').open() as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                if (set(row) != {'shard', 'grasp_index'} or type(row['grasp_index']) is not int
                        or row['grasp_index'] < 0):
                    raise ValueError('invalid training split row')
                shard = row['shard']
                if shard not in shards:
                    contained(root, shard)
                    shards[shard] = len(shards)
                shard_ids.append(shards[shard]); indices.append(row['grasp_index'])
        if not indices:
            raise ValueError('empty training split')
        self.shards = tuple(shards)
        self.shard_ids = np.asarray(shard_ids)
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return dict(shard=self.shards[self.shard_ids[index]], grasp_index=int(self.indices[index]))


def checkpoint_identity_matches(saved, current, root):
    """Accept only an exact identity or the archived, lossless val -> test migration.

    Shards, calibration, training rows and all other identity fields stay exact.
    Historical checkpoints keep their original three-way provenance.
    """
    if saved == current:
        return True
    archive = Path(root) / 'split_history/three_way_v1'
    if not archive.is_dir():
        return False
    try:
        legacy = {s: [json.loads(line) for line in (archive / f'{s}.jsonl').read_text().splitlines()
                      if line.strip()] for s in ('train', 'val', 'test')}
        expected = dict(current, samples={s: len(rows) for s, rows in legacy.items()},
                        split_sha256={s: sha256(archive / f'{s}.jsonl') for s in legacy})
        if saved != expected or legacy['train'] != read_split(root, 'train'):
            return False
        merged = [(r['shard'], r['grasp_index']) for s in ('val', 'test') for r in legacy[s]]
        actual = [(r['shard'], r['grasp_index']) for r in read_split(root, 'test')]
        return len(merged) == len(set(merged)) and sorted(merged) == sorted(actual)
    except (OSError, ValueError, KeyError):
        return False


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


def validate_grasp(grasp, ray_count, direction_axis=(0., 0., 1.)):
    for key in ('R', 't', 'q', 'K', 'd', 'h', 'hit_ratio'):
        x = np.asarray(grasp[key])
        if x.shape != FLOAT_SHAPES[key] or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError(f'invalid {key}: expected finite float32 {FLOAT_SHAPES[key]}')
    rotation = grasp['R']
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=2e-5):
        raise ValueError('R must be a proper rotation')
    if not np.allclose(grasp['d'], rotation @ np.asarray(direction_axis), atol=2e-5):
        raise ValueError('d must equal R times the versioned hand direction axis')
    if grasp['h'][0] < 0 or not 0 <= grasp['hit_ratio'] <= 1:
        raise ValueError('invalid gap or hit ratio')
    if ray_count < 64 or float(grasp['hit_ratio']) * ray_count < 64 - 1e-4:
        raise ValueError('palm_gap needs at least 64 hits')
    if not text(grasp['grasp_id']):
        raise ValueError('missing grasp_id')


class GraspDataset(Dataset):
    obs_keys = OBS_KEYS
    grasp_fields = {k: k for k in GRASP_KEYS}
    depth_key = 'palm_gap'
    default_conditions = ('object_points', 'scene_points', 'scene_mask', 'palm_gap', 'grasp_direction_cmd')

    def __init__(self, root, split, coord_scale_m, *, observation_cache_size=64, max_open_shards=8,
                 condition_keys=None, training_only=False, batched_tensors=False):
        self.root, self.split = Path(root).resolve(), split
        self.training_only = training_only
        self.batched_tensors = batched_tensors
        if batched_tensors and not training_only:
            raise ValueError('batched_tensors is for training-only rows')
        self.condition_keys = self.default_conditions if condition_keys is None else tuple(condition_keys)
        if ('object_points' not in self.condition_keys
                or set(self.condition_keys) - set(self.default_conditions)):
            raise ValueError('invalid dataset condition keys')
        if not np.isfinite(coord_scale_m) or coord_scale_m <= 0:
            raise ValueError('coordinate scale must be positive')
        if observation_cache_size < 0 or max_open_shards < 1:
            raise ValueError('invalid cache size')
        self.scale = float(coord_scale_m)
        self.rows = TrainingRows(root, split) if training_only else read_split(root, split)
        spec_path = self.root / 'hand/hand_spec.json'
        self.spec_hash = sha256(spec_path)
        spec = json.loads(spec_path.read_text())
        self._set_calibration(spec)
        self.cache_size, self.max_open = observation_cache_size, max_open_shards
        self._pid, self._handles, self._observations = os.getpid(), OrderedDict(), OrderedDict()

    def _set_calibration(self, spec):
        self.ray_count = len(spec['palm']['ray_origins_m'])
        self.direction_axis = direction_definition(spec)['local_axis']

    def _check_shard(self, f):
        if (f.attrs.get('schema_version') != 1 or f.attrs.get('hand_spec_sha256') != self.spec_hash
                or f.attrs.get('point_sampling') != 'full_mesh_area_uniform'):
            raise ValueError('shard schema, hand identity or sampling protocol mismatch')

    def _observation(self, f, oi):
        obs = {k: f[f'observations/{k}'][oi] for k in self.obs_keys}
        validate_observation(obs)
        return obs

    def _check_grasp(self, grasp):
        validate_grasp(grasp, self.ray_count, self.direction_axis)

    def object_id(self, f, oi):
        return text(f['observations/object_id'][oi])

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
            try:
                self._check_shard(f)
            except BaseException:
                f.close()
                raise
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
        grasp = {k: f[f'grasps/{field}'][gi] for k, field in self.grasp_fields.items()}
        oi = int(grasp['observation_index'])
        if oi < 0 or oi >= len(f['observations/object_points']):
            raise ValueError('observation index out of bounds')
        key = (row['shard'], oi)
        if key not in self._observations:
            obs = self._observation(f, oi)
            if self.cache_size:
                self._observations[key] = obs
                if len(self._observations) > self.cache_size:
                    self._observations.popitem(last=False)
        else:
            self._observations.move_to_end(key)
            obs = self._observations[key]
        self._check_grasp(grasp)
        return obs, grasp

    def __getitem__(self, index):
        obs, grasp = self.raw(index)
        center = obs['object_points'].mean(0, dtype=np.float64).astype(np.float32)
        condition = {k: torch.from_numpy((obs[k] - center) / self.scale)
                     for k in ('object_points', 'scene_points') if k in self.condition_keys}
        if 'scene_mask' in self.condition_keys:
            condition['scene_mask'] = torch.from_numpy(obs['scene_mask'].astype(bool))
        if self.depth_key in self.condition_keys:
            condition[self.depth_key] = torch.from_numpy(grasp['h'].copy())
        if 'grasp_direction_cmd' in self.condition_keys:
            condition['grasp_direction_cmd'] = torch.from_numpy(grasp['d'].copy())
        labels = {'h32': torch.from_numpy(grasp['h'].copy())} if self.depth_key == 'h32' else {}
        return dict(condition=condition, **labels,
                    keypoints_gt=torch.from_numpy((grasp['K'] - center) / self.scale),
                    metadata=dict(center_m=torch.from_numpy(center), coord_scale_m=self.scale,
                                  scene_id=text(obs['scene_id']), view_id=text(obs['view_id']),
                                  object_id=text(obs['object_id']), grasp_id=text(grasp['grasp_id']),
                                  shard=self.rows[index]['shard'], grasp_index=self.rows[index]['grasp_index'],
                                  split=self.split))

    def __getitems__(self, indices):
        """Batch HDF5 reads after the runner's deep audit, without unused labels.

        The public/evaluation path retains full metadata and per-grasp checks.
        Sorting unique HDF5 indices preserves arbitrary sampler order and repeats.
        """
        if not self.training_only:
            return [self[index] for index in indices]
        groups = {}
        for position, index in enumerate(indices):
            row = self.rows[index]
            groups.setdefault(row['shard'], []).append((position, row['grasp_index']))
        columns = {}
        result = [] if self.batched_tensors else [None] * len(indices)
        for shard, entries in groups.items():
            f = self._file(shard)
            unique, inverse = np.unique([gi for _, gi in entries], return_inverse=True)
            observation = f['grasps/observation_index'][unique]
            obs_unique, obs_inverse = np.unique(observation, return_inverse=True)
            points = f['observations/object_points'][obs_unique]
            centers = points.mean(1, dtype=np.float64).astype(np.float32)
            clouds = {'object_points': (points-centers[:, None])/self.scale}
            if 'scene_points' in self.condition_keys:
                clouds['scene_points'] = (f['observations/scene_points'][obs_unique]-centers[:, None])/self.scale
            if 'scene_mask' in self.condition_keys:
                clouds['scene_mask'] = f['observations/scene_mask'][obs_unique].astype(bool)
            grasps = f['grasps']
            targets = grasps[self.grasp_fields['K']][unique]
            origin = centers[obs_inverse].reshape(len(unique), *([1] * (targets.ndim - 2)), 3)
            targets = (targets - origin) / self.scale
            directions = grasps[self.grasp_fields['d']][unique] if 'grasp_direction_cmd' in self.condition_keys else None
            gaps = grasps[self.grasp_fields['h']][unique] if self.depth_key in self.condition_keys else None
            if not np.isfinite(targets).all() or (directions is not None and not np.isfinite(directions).all()):
                raise ValueError('nonfinite training targets/directions')
            if self.batched_tensors:
                # Gather once per column instead of creating and collating B Python dicts.
                positions = np.fromiter((pos for pos, _ in entries), dtype=np.intp)
                fields = {k: (v, obs_inverse[inverse]) for k, v in clouds.items()}
                fields['keypoints_gt'] = (targets, inverse)
                if directions is not None:
                    fields['grasp_direction_cmd'] = (directions, inverse)
                if gaps is not None:
                    fields[self.depth_key] = (gaps, inverse)
                for name, (values, rows) in fields.items():
                    if name not in columns:
                        columns[name] = np.empty((len(indices), *values.shape[1:]), dtype=values.dtype)
                    columns[name][positions] = values[rows]
                continue
            for (position, _), gi in zip(entries, inverse):
                oi = obs_inverse[gi]
                condition = {k: torch.from_numpy(v[oi]) for k, v in clouds.items()}
                if directions is not None:
                    condition['grasp_direction_cmd'] = torch.from_numpy(directions[gi])
                if gaps is not None:
                    condition[self.depth_key] = torch.from_numpy(gaps[gi])
                result[position] = dict(condition=condition, keypoints_gt=torch.from_numpy(targets[gi]))
        if self.batched_tensors:
            target = torch.from_numpy(columns.pop('keypoints_gt'))
            return dict(condition={k: torch.from_numpy(v) for k, v in columns.items()}, keypoints_gt=target)
        return result


def hdf5_dataset(dataset_format):
    if dataset_format == 'data_v2':
        return ThreeStageDataset
    if dataset_format == 'data_v1':
        return GraspDataset
    raise ValueError(f'expected HDF5 shards, got {dataset_format}')


def check_stage_order(value):
    if isinstance(value, (bytes, str)):
        value = json.loads(text(value))
    if value is None or tuple(map(text, value)) != STAGE_ORDER:
        raise ValueError(f'stage_order must be {STAGE_ORDER}')


class ThreeStageDataset(GraspDataset):
    """One observation and one candidate row supply all three control targets."""
    obs_keys = ('object_points', 'scene_id', 'view_id')
    grasp_fields = dict(observation_index='observation_index', grasp_id='candidate_id',
                        R='R', t='t', q='q', K='keypoints_gt', d='grasp_direction_cmd', h='h32')
    depth_key = 'h32'
    default_conditions = ('object_points', 'grasp_direction_cmd', 'h32')

    def _set_calibration(self, spec):
        self.direction_axis = official_wrist_definition(spec)['local_axis']
        self._scene_objects = {}

    def _check_shard(self, f):
        check_stage_order(f.attrs.get('stage_order'))
        if f.attrs.get('schema_version', 2) != 2 or f.attrs.get('hand_spec_sha256', self.spec_hash) != self.spec_hash:
            raise ValueError('data_v2 shard version or hand calibration mismatch')
        for key, shape in [('observations/object_points', (1024, 3)), ('grasps/R', (3, 3, 3)),
                           ('grasps/t', (3, 3)), ('grasps/q', (3, 22)), ('grasps/keypoints_gt', (3, 21, 3)),
                           ('grasps/grasp_direction_cmd', (3,)), ('grasps/h32', (1,))]:
            if f[key].shape[1:] != shape or f[key].dtype != np.float32:
                raise ValueError(f'{key} must contain float32 rows of shape {shape}')
        indices = f['grasps/observation_index']
        if indices.ndim != 1 or indices.dtype != np.int64:
            raise ValueError('observation_index must contain scalar int64 references')

    def object_id(self, f, oi):
        if 'object_id' in f['observations']:
            return super().object_id(f, oi)
        scene_id = text(f['observations/scene_id'][oi])
        if scene_id not in self._scene_objects:
            scene = json.loads(contained(self.root / 'scenes', scene_id + '.json').read_text())
            self._scene_objects[scene_id] = scene['object_id']
        return self._scene_objects[scene_id]

    def _observation(self, f, oi):
        obs = {k: f[f'observations/{k}'][oi] for k in self.obs_keys}
        obs['object_id'] = self.object_id(f, oi)
        if not np.isfinite(obs['object_points']).all() or not all(text(obs[k]) for k in ('object_id', 'scene_id', 'view_id')):
            raise ValueError('invalid three-stage observation')
        return obs

    def _check_grasp(self, grasp):
        if not all(np.isfinite(grasp[k]).all() for k in ('R', 't', 'q', 'K', 'h', 'd')):
            raise ValueError('nonfinite three-stage labels')
        r = grasp['R']
        if (not np.allclose(r.swapaxes(-1, -2) @ r, np.eye(3), atol=2e-5)
                or not np.allclose(np.linalg.det(r), 1, atol=2e-5)):
            raise ValueError('all three wrist rotations must be proper')
        if not np.allclose(grasp['d'], r[STAGE_ORDER.index('grasp')] @ self.direction_axis, atol=2e-5):
            raise ValueError('direction must match the grasp-frame wrist axis')
        if grasp['h'][0] < 0 or not text(grasp['grasp_id']):
            raise ValueError('invalid h32 or candidate_id')


def official_test_objects(root):
    objects, digests = set(), {}
    for release in ('DGN_5k', 'objaverse_5k'):
        path = Path(root) / release / 'valid_split/test.json'
        ids = json.loads(path.read_text())
        if not isinstance(ids, list) or not ids or not all(isinstance(s, str) and s for s in ids):
            raise ValueError(f'invalid official test object list in {path}')
        objects.update(ids)
        digests[release] = sha256(path)
    return objects, digests


def audit_dataset(root, coord_scale_m=0.2, *, deep=True, dataset_format='data_v1', official_test_root=None):
    """Full split leakage audit. Deep mode validates every referenced grasp, not a sample."""
    root = Path(root).resolve()
    hand = None
    if deep:
        from ..geometry.hand import URDFHand
        hand = URDFHand(root / 'hand/hand_spec.json')
    objects, scenes, references, grasp_ids = {}, {}, {}, {}
    counts, files = {}, {}
    scene_files = {}
    for split in SPLITS:
        ds = hdf5_dataset(dataset_format)(root, split, coord_scale_m, observation_cache_size=8)
        try:
            counts[split] = len(ds)
            for i, row in enumerate(ds.rows):
                f = ds._file(row['shard'])
                for group, keys in (('observations', ds.obs_keys), ('grasps', ds.grasp_fields.values())):
                    sizes = {len(f[f'{group}/{k}']) for k in keys}
                    if len(sizes) != 1:
                        raise ValueError(f'{group} dataset lengths disagree')
                if f['grasps/observation_index'].dtype != np.int64:
                    raise ValueError('observation_index must be int64')
                gi = row['grasp_index']
                oi = int(f['grasps/observation_index'][gi])
                if not 0 <= oi < len(f['observations/scene_id']):
                    raise ValueError('observation index out of bounds')
                oid = ds.object_id(f, oi)
                sid = text(f['observations/scene_id'][oi])
                gid = (sid, text(f['observations/view_id'][oi]), text(f['grasps'][ds.grasp_fields['grasp_id']][gi]))
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
                    for r, t, q, k in zip(grasp['R'].reshape(-1, 3, 3), grasp['t'].reshape(-1, 3),
                                          grasp['q'].reshape(-1, 22), grasp['K'].reshape(-1, 21, 3)):
                        if not hand.in_limits(q):
                            raise ValueError('joint angles outside calibrated hand limits')
                        if not np.allclose(hand.world_keypoints(r, t, q), k, atol=2e-6, rtol=1e-5):
                            raise ValueError('stored K disagrees with calibrated FK(R,t,q)')
                path = contained(root, row['shard'])
                if row['shard'] not in files:
                    stat = path.stat()
                    files[row['shard']] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
                    if deep:
                        files[row['shard']]['sha256'] = sha256(path)
            if dataset_format == 'data_v2':
                for sid in ds._scene_objects:
                    path = contained(root/'scenes', sid + '.json')
                    scene_files[str(path.relative_to(root))] = sha256(path)
        finally:
            ds.close()
    identity = dict(schema_version=1, samples=counts, objects=len(objects), scenes=len(scenes),
                    hand_spec_sha256=sha256(root / 'hand/hand_spec.json'),
                    split_sha256={s: sha256(root / 'splits' / f'{s}.jsonl') for s in SPLITS},
                    shard_stats=files, deep=deep)
    if dataset_format == 'data_v2':
        identity.update(schema_version=2, stage_order=list(STAGE_ORDER), scene_object_files=scene_files)
    else:
        identity['sampling'] = 'full_mesh_area_uniform'
    if official_test_root:
        excluded, identity['official_test_sha256'] = official_test_objects(official_test_root)
        overlap = {oid for oid, split in objects.items() if split == 'train'} & excluded
        if overlap:
            raise ValueError(f'official test objects in training split: {sorted(overlap)[:5]}')
    return identity
