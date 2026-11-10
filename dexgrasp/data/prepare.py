"""Build Data v1 shards from explicitly validated raw scene/grasp records."""
import json
from pathlib import Path
import shutil
import tempfile
import h5py
import numpy as np
from .dataset import (FLOAT_SHAPES, SPLITS, contained, sha256, validate_grasp,
                      validate_observation)
from ..geometry.hand import URDFHand
from ..geometry.mesh import (combine_meshes, load_mesh, measure_palm_gap,
                             sample_observation, validate_closed_mesh)


class ShardWriter:
    def __init__(self, path, hand_spec_path):
        self.file = h5py.File(path, 'x')
        spec = json.loads(Path(hand_spec_path).read_text())
        self.ray_count = len(spec['palm']['ray_origins_m'])
        self.file.attrs.update(schema_version=1, hand_spec_sha256=sha256(hand_spec_path),
                               point_sampling='full_mesh_area_uniform', label_version=spec['label_version'])
        self.obs_count, self.grasp_count = 0, 0
        for group, keys in (
            ('observations', ['object_points', 'scene_points', 'scene_mask', 'scene_id', 'view_id', 'object_id']),
            ('grasps', ['observation_index', 'grasp_id', 'R', 't', 'q', 'K', 'd', 'h', 'hit_ratio']),
        ):
            g = self.file.create_group(group)
            for key in keys:
                if key in FLOAT_SHAPES:
                    shape, dtype = FLOAT_SHAPES[key], np.float32
                elif key == 'scene_mask':
                    shape, dtype = (2048,), np.uint8
                elif key == 'observation_index':
                    shape, dtype = (), np.int64
                else:
                    shape, dtype = (), h5py.string_dtype('utf-8')
                options = dict(compression='lzf', shuffle=True) if shape else {}
                g.create_dataset(key, shape=(0, *shape), maxshape=(None, *shape),
                                 chunks=(1, *shape) if shape else (256,), dtype=dtype, **options)

    def _append(self, group, row, index):
        for key, ds in self.file[group].items():
            ds.resize(index+1, axis=0)
            ds[index] = row[key]

    def observation(self, row):
        validate_observation(row)
        index = self.obs_count
        self._append('observations', row, index)
        self.obs_count += 1
        return index

    def grasp(self, row):
        validate_grasp(row, self.ray_count)
        if not 0 <= row['observation_index'] < self.obs_count:
            raise ValueError('grasp references an absent observation')
        index = self.grasp_count
        self._append('grasps', row, index)
        self.grasp_count += 1
        return index

    def close(self):
        self.file.close()


def scene_meshes(root, scene):
    meshes, instances = {}, scene['instances']
    for instance in instances:
        key = instance['instance_id']
        if key in meshes:
            raise ValueError('duplicate scene instance id')
        v, f = load_mesh(contained(root, instance['mesh']))
        scale = np.asarray(instance['scale'], float)
        if scale.shape not in ((), (3,)) or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('instance scale must be positive scalar or xyz')
        rotation, translation = np.asarray(instance['R'], float), np.asarray(instance['t'], float)
        if (rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(translation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5)
                or not np.isclose(np.linalg.det(rotation), 1, atol=2e-5)):
            raise ValueError('invalid instance scene pose')
        meshes[key] = ((v*scale) @ rotation.T+translation, f)
    target_id = scene['target_id']
    if target_id not in meshes:
        raise ValueError('target instance not present')
    target = meshes[target_id]
    environment = combine_meshes([mesh for key, mesh in meshes.items() if key != target_id])
    return target, environment


def prepare_dataset(root, *, seed=0, grasps_per_shard=4096):
    root = Path(root).resolve()
    if grasps_per_shard < 1:
        raise ValueError('grasps_per_shard must be positive')
    for name in ('shards', 'splits', 'preparation.json', 'invalid_labels.jsonl'):
        if (root/name).exists():
            raise FileExistsError(f'{root/name} already exists; choose a new dataset version')
    hand = URDFHand(root/'hand/hand_spec.json')
    scene_paths = sorted((root/'scenes').glob('*.json'))
    if not scene_paths:
        raise ValueError('no raw scenes found')
    scene_specs = [(p, json.loads(p.read_text())) for p in scene_paths]
    object_splits, scene_splits = {}, {}
    for _, scene in scene_specs:
        if scene['split'] not in SPLITS:
            raise ValueError('each scene needs an explicit train/val/test split')
        for mapping, key in ((object_splits, scene['object_id']), (scene_splits, scene['scene_id'])):
            if key in mapping and mapping[key] != scene['split']:
                raise ValueError('object/scene split leakage in raw input')
            mapping[key] = scene['split']
        if not scene.get('collision_validation_version'):
            raise ValueError('raw scene must record its external grasp collision-validation protocol')
    staging = Path(tempfile.mkdtemp(prefix='.prepare-', dir=root))
    (staging/'shards').mkdir()
    (staging/'splits').mkdir()
    rows, rejected, sources = {s: [] for s in SPLITS}, [], {}
    rng, writer, shard_index = np.random.default_rng(seed), None, -1
    seen_obs, seen_grasps = set(), set()
    try:
        for path, scene in scene_specs:
            observation_id = (scene['scene_id'], scene['view_id'])
            if observation_id in seen_obs:
                raise ValueError('duplicate scene/view observation')
            seen_obs.add(observation_id)
            target, environment = scene_meshes(root, scene)
            validate_closed_mesh(*target)
            obs = sample_observation(target, environment, rng)
            obs.update({k: scene[k] for k in ('scene_id', 'view_id', 'object_id')})
            sources[str(path.relative_to(root))] = sha256(path)
            grasp_path = contained(root, scene['grasps'])
            sources[str(grasp_path.relative_to(root))] = sha256(grasp_path)
            for instance in scene['instances']:
                mesh_path = contained(root, instance['mesh'])
                sources[str(mesh_path.relative_to(root))] = sha256(mesh_path)
            obs_index = None
            with np.load(grasp_path, allow_pickle=False) as grasps:
                count = len(grasps['R'])
                if any(len(grasps[k]) != count for k in ('t', 'q', 'valid', 'grasp_id')):
                    raise ValueError('raw grasp field lengths disagree')
                if grasps['valid'].dtype != bool:
                    raise ValueError('raw valid must be a boolean external-validation result')
                for i in range(count):
                    gid = str(grasps['grasp_id'][i])
                    identity = (*observation_id, gid)
                    if identity in seen_grasps:
                        raise ValueError('duplicate raw grasp identity')
                    seen_grasps.add(identity)
                    rotation, translation, q = (np.asarray(grasps[k][i], np.float32) for k in ('R', 't', 'q'))
                    if not grasps['valid'][i]:
                        rejected.append(dict(scene_id=scene['scene_id'], grasp_id=gid, reason='external_grasp_invalid'))
                        continue
                    if q.shape != (22,) or not np.isfinite(q).all() or not hand.in_limits(q):
                        raise ValueError('raw joint labels are invalid or outside Sharpa limits')
                    if (rotation.shape != (3, 3) or translation.shape != (3,)
                            or not np.isfinite(translation).all()
                            or not np.allclose(rotation.T@rotation, np.eye(3), atol=2e-5)
                            or not np.isclose(np.linalg.det(rotation), 1, atol=2e-5)):
                        raise ValueError('invalid raw grasp root pose')
                    gap = measure_palm_gap(*target, rotation, translation, hand.spec['palm'], check_mesh=False)
                    if not gap['valid']:
                        rejected.append(dict(scene_id=scene['scene_id'], grasp_id=gid, **gap))
                        continue
                    if writer is None or writer.grasp_count >= grasps_per_shard:
                        if writer is not None:
                            writer.close()
                        shard_index += 1
                        shard = f'shards/part-{shard_index:05d}.h5'
                        writer = ShardWriter(staging/shard, hand.spec_path)
                        obs_index = None
                    if obs_index is None:
                        obs_index = writer.observation(obs)
                    row = dict(observation_index=obs_index, grasp_id=gid, R=rotation, t=translation, q=q,
                               K=hand.world_keypoints(rotation, translation, q).astype(np.float32),
                               d=rotation[:, 2].copy(), h=np.array([gap['h']], np.float32),
                               hit_ratio=np.float32(gap['hit_ratio']))
                    gi = writer.grasp(row)
                    rows[scene['split']].append(dict(shard=shard, grasp_index=gi))
        if any(not rows[s] for s in SPLITS):
            raise ValueError('each split needs at least one valid grasp after labelling')
        if writer is not None:
            writer.close()
            writer = None
        for split, split_rows in rows.items():
            (staging/'splits'/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in split_rows))
        report = dict(schema_version=1, seed=seed, sampling='full_mesh_area_uniform',
                      samples={s: len(r) for s, r in rows.items()}, rejected=len(rejected),
                      sources_sha256=sources, hand_spec_sha256=sha256(hand.spec_path),
                      hand_urdf_sha256=sha256(hand.urdf_path),
                      note='Raw valid flags are supplied externally; preparation does not simulate grasps.')
        (staging/'preparation.json').write_text(json.dumps(report, indent=2)+'\n')
        (staging/'invalid_labels.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rejected))
        for name in ('shards', 'splits', 'preparation.json', 'invalid_labels.jsonl'):
            (staging/name).rename(root/name)
        return report
    finally:
        if writer is not None:
            writer.close()
        shutil.rmtree(staging)
