"""Strict configuration for the user's v1, with no baseline fallbacks."""
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


MODEL_VARIANTS = (
    'full', 'full_h_less', 'object_only', 'object_only_h_less',
    'object_direction_adaln', 'object_direction_adaln_h_less',
)


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
    variant: str = 'full'
    self_attention_backend: str = 'auto'

    @property
    def use_scene(self):
        return self.variant in ('full', 'full_h_less')

    @property
    def use_gap(self):
        return not self.variant.endswith('_h_less')

    @property
    def direction_adaln(self):
        return self.variant.startswith('object_direction_adaln')

    @property
    def condition_keys(self):
        keys = ['object_points', 'grasp_direction_cmd']
        if self.use_scene:
            keys.extend(('scene_points', 'scene_mask'))
        if self.use_gap:
            keys.append('palm_gap')
        return tuple(keys)

    def __post_init__(self):
        if self.self_attention_backend not in ('auto', 'centered_fp32'):
            raise ValueError('unknown self-attention backend')
        if self.variant not in MODEL_VARIANTS:
            raise ValueError(f'unknown model variant: {self.variant!r}')
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
    num_workers=2, precision='float32', save_every=1000, val_every=250,
    val_batches=8, sample_steps=10, teacher_steps=32, teacher_solver='heun',
    full_interval_probability=0.5, observation_cache_size=64, max_open_shards=8,
    initialization='random', freeze_encoders=False, overfit_diagnostic=False,
    dataset_format='data_v1', data_loading='stream', prefetch_factor=2, cpu_threads=4, optimizer='adamw',
    muon_lr=0.02, muon_momentum=0.95, muon_ns_steps=5,
    fps_backend='torch', compile_transformer=False,
    sparse_convolution_mode='default',
    checkpoint_direction_bias=False,
    select_tokens_before_slice=False,
    lr_schedule='constant', warmup_steps=5000, warmup_start_factor=0.2, lr_min=2e-5,
    abort_grad_norm=0.0,
)


def load_config(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    allowed = set(DEFAULT_RUN) | {'model', 'dataset_root', 'hand_spec', 'run_dir',
                                 'checkpoint_dir', 'teacher_checkpoint', 'initial_checkpoint', 'data_audit'}
    if set(raw) - allowed:
        raise ValueError(f'unknown config keys: {sorted(set(raw) - allowed)}')
    cfg = {**DEFAULT_RUN, **raw}
    if cfg['lr_schedule'] not in ('constant', 'warmup_cosine'):
        raise ValueError('unsupported lr_schedule')
    if cfg['lr_schedule'] == 'warmup_cosine':
        if (type(cfg['warmup_steps']) is not int or not 0 < cfg['warmup_steps'] < cfg['max_steps']
                or not 0 < cfg['warmup_start_factor'] <= 1 or not 0 <= cfg['lr_min'] <= cfg['lr']):
            raise ValueError('invalid warmup/cosine settings')
    if cfg['schema_version'] != 1 or cfg['method'] not in ('fm', 'flowmap'):
        raise ValueError('expected v1 fm or flowmap configuration')
    if cfg['initialization'] not in ('random', 'fm_teacher', 'checkpoint'):
        raise ValueError('initialization must be random, fm_teacher or checkpoint')
    if cfg['initialization'] == 'fm_teacher' and cfg['method'] != 'flowmap':
        raise ValueError('fm_teacher initialization is only for FlowMap')
    for key in ('freeze_encoders', 'overfit_diagnostic', 'compile_transformer', 'checkpoint_direction_bias',
                'select_tokens_before_slice'):
        if type(cfg[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    for key in ('batch_size', 'accumulation', 'max_steps', 'save_every', 'val_every',
                'val_batches', 'sample_steps', 'teacher_steps', 'max_open_shards',
                'prefetch_factor', 'cpu_threads', 'muon_ns_steps'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('num_workers', 'observation_cache_size', 'seed'):
        if type(cfg[key]) is not int or not 0 <= cfg[key] < 2**32:
            raise ValueError(f'{key} must be a nonnegative integer')
    for key in ('lr', 'weight_decay', 'grad_clip', 'full_interval_probability', 'muon_lr', 'muon_momentum', 'abort_grad_norm'):
        if not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]):
            raise ValueError(f'{key} must be finite')
    if cfg['dataset_format'] not in ('data_v1', 'object_corpus', 'packed'):
        raise ValueError('unsupported dataset_format')
    if cfg['data_loading'] not in ('stream', 'gpu_cache'):
        raise ValueError('data_loading must be stream or gpu_cache')
    if cfg['fps_backend'] not in ('torch', 'triton'):
        raise ValueError('fps_backend must be torch or triton')
    if cfg['sparse_convolution_mode'] not in ('default', 'direct', 'copy', 'copy_hidden'):
        raise ValueError('sparse_convolution_mode must be default, direct, copy or copy_hidden')
    if cfg['optimizer'] not in ('adamw', 'muon') or cfg['muon_lr'] <= 0 or not 0 <= cfg['muon_momentum'] < 1:
        raise ValueError('invalid optimizer or Muon settings')
    if cfg['muon_ns_steps'] >= 100:
        raise ValueError('muon_ns_steps must be below 100')
    if cfg['lr'] <= 0 or cfg['grad_clip'] <= 0 or cfg['weight_decay'] < 0:
        raise ValueError('invalid optimizer settings')
    if cfg['abort_grad_norm'] < 0:
        raise ValueError('abort_grad_norm must be nonnegative (zero disables)')
    if not 0 < cfg['full_interval_probability'] <= 1:
        raise ValueError('full interval probability must be in (0, 1]')
    if cfg['precision'] not in ('float32', 'bfloat16') or cfg['teacher_solver'] not in ('euler', 'heun'):
        raise ValueError('unsupported precision or ODE solver')
    model = cfg.get('model', {})
    if set(model) - {f.name for f in fields(ModelConfig)}:
        raise ValueError('unknown model config keys')
    if model.get('variant', 'full') not in MODEL_VARIANTS:
        raise ValueError('unknown model variant')
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
    if cfg.get('data_audit'):
        cfg['data_audit'] = str((path.parent / cfg['data_audit']).resolve())
    return cfg


def model_config(cfg, hand_spec):
    return ModelConfig(**cfg['model'], hand_reference_length_m=hand_spec['reference_length_m'])
