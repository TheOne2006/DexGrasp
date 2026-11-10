"""Strict configuration for the user's v1, with no baseline fallbacks."""
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    width: int = 512
    depth: int = 8
    heads: int = 8
    tokens: int = 128
    backbone_channels: int = 512
    keypoints: int = 21
    voxel_size_m: float = 0.005
    coord_scale_m: float = 0.2
    hand_reference_length_m: float = 0.0  # Must come from a calibrated hand spec.

    def __post_init__(self):
        for key in ('width', 'depth', 'heads', 'tokens', 'backbone_channels', 'keypoints'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.width % self.heads or self.keypoints != 21 or self.tokens > 1024:
            raise ValueError('invalid attention dimensions or Sharpa keypoint count')
        for key in ('voxel_size_m', 'coord_scale_m', 'hand_reference_length_m'):
            value = getattr(self, key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be calibrated, finite and positive')


DEFAULT_RUN = dict(
    schema_version=1, method='fm', seed=0, device='cuda:0', batch_size=8,
    accumulation=1, max_steps=10000, lr=2e-4, weight_decay=0.01, grad_clip=1.0,
    num_workers=2, precision='float32', save_every=500, val_every=250,
    val_batches=8, sample_steps=10, teacher_steps=32, teacher_solver='heun',
    full_interval_probability=0.5, observation_cache_size=64, max_open_shards=8,
    initialization='random', freeze_encoders=False, overfit_diagnostic=False,
)


def load_config(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    allowed = set(DEFAULT_RUN) | {'model', 'dataset_root', 'hand_spec', 'run_dir',
                                 'checkpoint_dir', 'teacher_checkpoint', 'initial_checkpoint'}
    if set(raw) - allowed:
        raise ValueError(f'unknown config keys: {sorted(set(raw) - allowed)}')
    cfg = {**DEFAULT_RUN, **raw}
    if cfg['schema_version'] != 1 or cfg['method'] not in ('fm', 'flowmap'):
        raise ValueError('expected v1 fm or flowmap configuration')
    if cfg['initialization'] not in ('random', 'fm_teacher', 'checkpoint'):
        raise ValueError('initialization must be random, fm_teacher or checkpoint')
    if cfg['initialization'] == 'fm_teacher' and cfg['method'] != 'flowmap':
        raise ValueError('fm_teacher initialization is only for FlowMap')
    for key in ('freeze_encoders', 'overfit_diagnostic'):
        if type(cfg[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    for key in ('batch_size', 'accumulation', 'max_steps', 'save_every', 'val_every',
                'val_batches', 'sample_steps', 'teacher_steps', 'max_open_shards'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('num_workers', 'observation_cache_size', 'seed'):
        if type(cfg[key]) is not int or not 0 <= cfg[key] < 2**32:
            raise ValueError(f'{key} must be a nonnegative integer')
    for key in ('lr', 'weight_decay', 'grad_clip', 'full_interval_probability'):
        if not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]):
            raise ValueError(f'{key} must be finite')
    if cfg['lr'] <= 0 or cfg['grad_clip'] <= 0 or cfg['weight_decay'] < 0:
        raise ValueError('invalid optimizer settings')
    if not 0 < cfg['full_interval_probability'] <= 1:
        raise ValueError('full interval probability must be in (0, 1]')
    if cfg['precision'] not in ('float32', 'bfloat16') or cfg['teacher_solver'] not in ('euler', 'heun'):
        raise ValueError('unsupported precision or ODE solver')
    model = cfg.get('model', {})
    if set(model) - {f.name for f in fields(ModelConfig)}:
        raise ValueError('unknown model config keys')
    # Hand length is loaded from hand_spec at plan/run time, never guessed here.
    if 'hand_reference_length_m' in model:
        raise ValueError('hand_reference_length_m belongs in the calibrated hand_spec')
    cfg['model'] = model
    for key in ('dataset_root', 'hand_spec', 'run_dir', 'checkpoint_dir'):
        if not isinstance(cfg.get(key), str) or not cfg[key]:
            raise ValueError(f'missing {key}')
        cfg[key] = str((path.parent / cfg[key]).resolve())
    if cfg['run_dir'] == cfg['checkpoint_dir']:
        raise ValueError('run and checkpoint directories must differ')
    teacher = cfg.get('teacher_checkpoint')
    if cfg['method'] == 'flowmap' and not teacher:
        raise ValueError('FlowMap requires a frozen FM teacher checkpoint')
    if teacher:
        cfg['teacher_checkpoint'] = str((path.parent / teacher).resolve())
    initial = cfg.get('initial_checkpoint')
    if (cfg['initialization'] == 'checkpoint') != bool(initial):
        raise ValueError('checkpoint initialization requires initial_checkpoint and vice versa')
    if initial:
        cfg['initial_checkpoint'] = str((path.parent / initial).resolve())
    return cfg


def model_config(cfg, hand_spec):
    return ModelConfig(**cfg['model'], hand_reference_length_m=hand_spec['reference_length_m'])
