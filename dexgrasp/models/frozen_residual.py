"""Frozen single-grasp FM with independently parameterized approach and squeeze flows."""
from dataclasses import dataclass, field
from functools import cached_property

import torch
from torch import nn
from .config import ModelConfig
from .execution import configure_keypoint_execution
from .keypoint import ConditionalBlock, KeypointModel, time_embedding
from .sampling import evaluation, repeat_context


@dataclass(frozen=True)
class FrozenResidualConfig:
    family: str = 'frozen_grasp_residual'
    grasp_model: dict = field(default_factory=dict)
    hand_reference_length_m: float = 0.0
    grasp_steps: int = 20
    checkpoint_schema = 'dexgrasp-frozen-residual-v1'
    state_shape = (3, 21, 3)
    stages = 3

    @cached_property
    def parent(self):
        return ModelConfig(**self.grasp_model, hand_reference_length_m=self.hand_reference_length_m)

    @property
    def coord_scale_m(self):
        return self.parent.coord_scale_m

    @property
    def condition_keys(self):
        return tuple('h32' if key == 'palm_gap' else key for key in self.parent.condition_keys)

    def __post_init__(self):
        if self.parent.stages != 1 or self.parent.use_scene or self.parent.target != 'absolute':
            raise ValueError('frozen grasp requires a single-pose object-conditioned FM parent')
        if type(self.grasp_steps) is not int or self.grasp_steps < 1:
            raise ValueError('grasp_steps must be positive')

    @staticmethod
    def validate_recipe(settings, dataset_format):
        if dataset_format != 'data_v2':
            raise ValueError('frozen residual training requires paired DataV2')


class ResidualFM(nn.Module):
    def __init__(self, parent_width):
        super().__init__()
        cfg = ModelConfig(width=256, heads=4, depth=2, conditioning={})
        self.state = nn.Linear(6, 256)
        self.grasp = nn.Linear(parent_width, 256)
        self.object = nn.Linear(parent_width, 256)
        self.ids = nn.Embedding(21, 256)
        self.time = nn.Sequential(nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 256))
        self.blocks = nn.ModuleList(ConditionalBlock(cfg, i, ffn_ratio=2) for i in range(2))
        self.head = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 3))

    def prepare(self, hidden, tokens, cache_kv):
        memory = (self.object(tokens),)
        kv = [block.prepare_kv(memory) for block in self.blocks] if cache_kv else [None, None]
        return self.grasp(hidden) + self.ids.weight, memory, kv

    def forward(self, state, time, grasp, features):
        aligned, memory, kv = features
        x = self.state(torch.cat((state, grasp), dim=-1)) + aligned
        t = self.time(time_embedding(time))
        for block, cache in zip(self.blocks, kv):
            x = block(x, t, memory, (), kv=cache)
        return self.head(x)


class FrozenResidualModel(nn.Module):
    def __init__(self, cfg, method='fm', **options):
        super().__init__()
        if method != 'fm':
            raise ValueError('dual residual heads use FM')
        self.cfg, self.method = cfg, method
        self.grasp = KeypointModel(cfg.parent, method, **options).requires_grad_(False).eval()
        self.pregrasp = ResidualFM(cfg.parent.width)
        self.squeeze = ResidualFM(cfg.parent.width)

    def train(self, mode=True):
        super().train(mode)
        self.grasp.eval()
        return self

    def cache_condition_indices(self, condition):
        self.grasp.cache_condition_indices(condition)

    @torch.no_grad()
    def encode(self, condition, *, cache_kv=False):
        if self.grasp.cfg.use_gap:
            condition = dict(condition, palm_gap=condition['h32'])
        return self.grasp.encode(condition, cache_kv=cache_kv)

    def training_loss(self, batch):
        from ..train.objectives import fm_pair
        pre, grasp, squeeze = batch['keypoints_gt'].unbind(1)
        with torch.no_grad():
            context = self.encode(batch['condition'])
            hidden = self.grasp.features(grasp, grasp.new_ones(len(grasp)), context)
        losses = []
        for flow, absolute in ((self.pregrasp, pre), (self.squeeze, squeeze)):
            state, time, target = fm_pair(absolute - grasp)
            features = flow.prepare(hidden, context.tokens[0], False)
            prediction = flow(state, time, grasp, features)
            losses.append((prediction.float() - target).square().mean())
        return (losses[0] + losses[1]) * .5


@torch.no_grad()
def sample_residual(model, condition, *, steps, candidates=1, noise=None, cache_kv=True):
    if any(type(i) is not int or i < 1 for i in (steps, candidates)):
        raise ValueError('positive integer steps and candidates required')
    points = condition['object_points']
    shape = (len(points)*candidates, 3, 21, 3)
    if noise is None:
        noise = torch.randn(shape, device=points.device)
    if noise.shape != shape or not torch.isfinite(noise).all():
        raise ValueError('invalid initial noise')
    pre, grasp, squeeze = noise.float().unbind(1)
    with evaluation(model):
        context = repeat_context(model.encode(condition, cache_kv=cache_kv), candidates)
        n = model.cfg.grasp_steps
        for i in range(n):
            t = grasp.new_full((len(grasp),), i/n)
            grasp = grasp + model.grasp(grasp, t, context).float()/n
        hidden = model.grasp.features(grasp, grasp.new_ones(len(grasp)), context)
        pre_context = model.pregrasp.prepare(hidden, context.tokens[0], cache_kv)
        squeeze_context = model.squeeze.prepare(hidden, context.tokens[0], cache_kv)
        for i in range(steps):
            t = grasp.new_full((len(grasp),), i/steps)
            pre = pre + model.pregrasp(pre, t, grasp, pre_context).float()/steps
            squeeze = squeeze + model.squeeze(squeeze, t, grasp, squeeze_context).float()/steps
    poses = torch.stack((grasp + pre, grasp, grasp + squeeze), dim=1)
    if not torch.isfinite(poses).all():
        raise FloatingPointError('nonfinite frozen residual output')
    return poses


def configure_residual_execution(model, cfg):
    configure_keypoint_execution(model.grasp, cfg)
    if cfg and cfg['compile_transformer']:
        for flow in (model.pregrasp, model.squeeze):
            for block in flow.blocks:
                block.compile(fullgraph=True, dynamic=False)
