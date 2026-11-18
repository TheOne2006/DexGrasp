"""Transformer architecture and conditioning. Independent of data loading and training."""
from dataclasses import dataclass
import json
import math
import torch
from pathlib import Path


MODEL_VARIANTS = (
    'full', 'full_h_less', 'object_only', 'object_only_h_less',
    'object_direction_adaln', 'object_direction_adaln_h_less',
)
STAGE_ORDER = ('pregrasp', 'grasp', 'squeeze')


@dataclass(frozen=True)
class BlockConditions:
    """Independent condition inputs to one Transformer block. Time is always present."""
    adaln: bool = True
    direction_adaln: bool = False
    geometry_bias: bool = False
    depth: bool = False

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) is not bool:
                raise ValueError(f'conditioning.{name} must be boolean')
        if self.direction_adaln and not self.adaln:
            raise ValueError('direction_adaln requires AdaLN in the same block')

    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict) or value.keys() - cls.__dataclass_fields__.keys():
            raise ValueError('unknown conditioning field or invalid block configuration')
        return cls(**value)


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
    variant: str | None = None
    self_attention_backend: str = 'auto'
    stages: int = 1
    conditioning: BlockConditions | tuple[BlockConditions, ...] | None = None
    family: str = 'keypoint_flow'
    target: str = 'absolute'  # 'grasp' selects the middle frame of DataV2.
    geometry_source: str = 'per_stage'
    geometry_bias_scale: float = 1.0

    def absolute_target(self, poses):
        return poses[:, 1] if self.target == 'grasp' else poses

    def training_state(self, poses):
        if self.target == 'residual':
            pre, grasp, squeeze = poses.unbind(dim=1)
            return torch.stack((pre - grasp, grasp, squeeze - grasp), dim=1)
        return self.absolute_target(poses)

    def absolute_state(self, state):
        if self.target == 'residual':
            pre, grasp, squeeze = state.unbind(dim=1)
            return torch.stack((pre + grasp, grasp, squeeze + grasp), dim=1)
        return state

    @property
    def state_shape(self):
        return (self.keypoints, 3) if self.stages == 1 else (self.stages, self.keypoints, 3)

    @property
    def checkpoint_schema(self):
        return 'dexgrasp-v2' if self.stages == 3 else 'dexgrasp-v1'

    @property
    def depth_key(self):
        return 'h32' if self.stages == 3 else 'palm_gap'

    @property
    def use_scene(self):
        return self.variant in ('full', 'full_h_less')

    @property
    def block_conditions(self):
        if isinstance(self.conditioning, tuple):
            return self.conditioning
        shared = self.conditioning
        if shared is None:
            # Historical recipes encoded mutually exclusive direction paths in their names.
            adaln = self.variant.startswith('object_direction_adaln')
            shared = BlockConditions(direction_adaln=adaln, geometry_bias=not adaln,
                                     depth=not self.variant.endswith('_h_less'))
        return (shared,) * self.depth

    @property
    def use_gap(self):
        return any(block.depth for block in self.block_conditions)

    @property
    def direction_adaln(self):
        return any(block.direction_adaln for block in self.block_conditions)

    @property
    def geometry_bias(self):
        return any(block.geometry_bias for block in self.block_conditions)

    @property
    def condition_keys(self):
        keys = ['object_points']
        if self.direction_adaln or self.geometry_bias:
            keys.append('grasp_direction_cmd')
        if self.use_scene:
            keys.extend(('scene_points', 'scene_mask'))
        if self.use_gap:
            keys.append(self.depth_key)
        return tuple(keys)

    def __post_init__(self):
        if self.family != 'keypoint_flow':
            raise ValueError(f'expected keypoint_flow model family, got {self.family!r}')
        if self.self_attention_backend not in ('auto', 'centered_fp32'):
            raise ValueError('unknown self-attention backend')
        if self.conditioning is None:
            if self.variant is None:
                object.__setattr__(self, 'variant', 'full')
            if self.variant not in MODEL_VARIANTS:
                raise ValueError(f'unknown model variant: {self.variant!r}')
        else:
            if self.variant not in (None, 'configured'):
                raise ValueError('choose either explicit conditioning or a legacy variant')
            if isinstance(self.conditioning, (tuple, list)):
                if len(self.conditioning) != self.depth:
                    raise ValueError('per-layer conditioning must contain one entry for each Transformer block')
                branches = tuple(BlockConditions.parse(item) for item in self.conditioning)
            else:
                branches = BlockConditions.parse(self.conditioning)
            object.__setattr__(self, 'conditioning', branches)
            object.__setattr__(self, 'variant', 'configured')
        if type(self.stages) is not int or self.stages not in (1, 3):
            raise ValueError('stages must be 1 or 3')
        if self.target not in ('grasp', 'absolute', 'residual'):
            raise ValueError('unknown training target')
        if self.target == 'grasp' and (self.stages != 1 or self.use_scene or self.use_gap):
            raise ValueError('DataV2 grasp-only baseline requires one stage, object only and no depth')
        if self.target == 'residual' and (self.stages != 3 or self.geometry_source != 'grasp'):
            raise ValueError('residual targets require three stages and grasp geometry')
        if self.geometry_source not in ('grasp', 'per_stage') or not math.isfinite(self.geometry_bias_scale):
            raise ValueError('invalid geometry bias configuration')
        if self.stages == 3 and self.use_scene:
            raise ValueError('three-stage generation requires an object-only encoder')
        for key in ('width', 'depth', 'heads', 'tokens', 'backbone_channels', 'keypoints'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.width % self.heads or self.keypoints != 21 or self.tokens > 1024:
            raise ValueError('invalid attention dimensions or Sharpa keypoint count')
        scales = ['voxel_size_m', 'coord_scale_m']
        if self.use_gap or self.hand_reference_length_m != 0:
            scales.append('hand_reference_length_m')
        for key in scales:
            value = getattr(self, key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be calibrated, finite and positive')

    @classmethod
    def from_file(cls, path, *, hand_reference_length_m=0.0):
        """Instantiate a model recipe independently of the training runner and dataset."""
        return cls(**read_model_settings(path), hand_reference_length_m=hand_reference_length_m)

    @staticmethod
    def validate_recipe(settings, dataset_format):
        variant = settings.get('variant')
        if settings.get('conditioning') is not None and variant not in (None, 'configured'):
            raise ValueError('choose either explicit conditioning or a legacy variant')
        if settings.get('conditioning') is None and variant is not None and variant not in MODEL_VARIANTS:
            raise ValueError('unknown model variant')
        if (dataset_format == 'data_v2') != (settings.get('stages', 1) == 3 or settings.get('target') == 'grasp'):
            raise ValueError('three-stage keypoint models require paired data_v2 shards and vice versa')


def read_model_settings(path):
    settings = json.loads(Path(path).read_text())
    if not isinstance(settings, dict):
        raise ValueError('expected a model-only configuration')
    if 'hand_reference_length_m' in settings:
        raise ValueError('hand_reference_length_m belongs in the calibrated hand_spec')
    return settings
