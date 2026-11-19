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
    parent_depth_label: str | None = None
    # Checkpoints written before branch dimensions were recorded used 2 x 256.
    residual_depth: int = 2
    residual_width: int = 256
    residual_heads: int = 4
    residual_self_attention_backend: str = 'auto'
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
        if self.parent.use_gap and self.parent_depth_label != 'h32':
            raise ValueError('confirm parent_depth_label=h32 for a depth-enabled parent trained on this label')
        if type(self.grasp_steps) is not int or self.grasp_steps < 1:
            raise ValueError('grasp_steps must be positive')

    @staticmethod
    def validate_recipe(settings, dataset_format):
        if dataset_format != 'data_v2':
            raise ValueError('frozen residual training requires paired DataV2')


class ResidualFM(nn.Module):
    def __init__(self, parent_width, cfg):
        super().__init__()
        width = cfg.width
        self.state = nn.Linear(6, width)
        self.grasp = nn.Linear(parent_width, width)
        self.object = nn.Linear(parent_width, width)
        self.ids = nn.Embedding(cfg.keypoints, width)
        self.time = nn.Sequential(nn.Linear(128, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList(ConditionalBlock(cfg, i, ffn_ratio=2) for i in range(cfg.depth))
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 128), nn.SiLU(), nn.Linear(128, 3))

    def prepare(self, hidden, tokens, cache_kv):
        memory = (self.object(tokens),)
        kv = [block.prepare_kv(memory) if cache_kv else None for block in self.blocks]
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
        branch = ModelConfig(depth=cfg.residual_depth, width=cfg.residual_width,
                             heads=cfg.residual_heads, conditioning={},
                             self_attention_backend=cfg.residual_self_attention_backend)
        self.grasp = KeypointModel(cfg.parent, method, **options).requires_grad_(False).eval()
        self.pregrasp = ResidualFM(cfg.parent.width, branch)
        self.squeeze = ResidualFM(cfg.parent.width, branch)

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


def integrate_residual(flow, state, grasp, hidden, tokens, steps, cache_kv):
    features = flow.prepare(hidden, tokens, cache_kv)
    for i in range(steps):
        time = grasp.new_full((len(grasp),), i/steps)
        state = state + flow(state, time, grasp, features).float()/steps
    return state


@torch.no_grad()
def sample_residual(model, condition, *, steps, candidates=1, noise=None, cache_kv=True,
                    parallel_branches=True):
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
        args = (grasp, hidden, context.tokens[0], steps, cache_kv)
        if parallel_branches and grasp.is_cuda:
            # Both branches read the same completed anchor and hidden features.
            # All side-stream use finishes before the caller can recycle those tensors.
            current = torch.cuda.current_stream(grasp.device)
            side = torch.cuda.Stream(device=grasp.device)
            side.wait_stream(current)
            try:
                with torch.cuda.stream(side):
                    squeeze = integrate_residual(model.squeeze, squeeze, *args)
                pre = integrate_residual(model.pregrasp, pre, *args)
            finally:
                current.wait_stream(side)
            squeeze.record_stream(current)
        else:
            pre = integrate_residual(model.pregrasp, pre, *args)
            squeeze = integrate_residual(model.squeeze, squeeze, *args)
    poses = torch.stack((grasp + pre, grasp, grasp + squeeze), dim=1)
    if not torch.isfinite(poses).all():
        raise FloatingPointError('nonfinite frozen residual output')
    return poses


def configure_residual_execution(model, cfg):
    configure_keypoint_execution(model.grasp, cfg)
    if model.cfg.residual_self_attention_backend == 'centered_fp32':
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
    if cfg and cfg['compile_transformer']:
        for flow in (model.pregrasp, model.squeeze):
            for block in flow.blocks:
                block.compile(fullgraph=True, dynamic=False)
