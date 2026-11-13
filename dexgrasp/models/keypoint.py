"""Shared v1 conditional Transformer; independent FM and FlowMap parameters."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .encoders import GeometryEncoder


def time_embedding(t, width=128):
    freq = torch.exp(-math.log(10000) * torch.arange(width // 2, device=t.device) / (width // 2))
    phase = t.float()[:, None] * freq[None] * 1000
    return torch.cat((phase.sin(), phase.cos()), -1)


def direction_geometry(state, centers, direction):
    """Compute once per forward, in FP32 even when attention uses BF16."""
    r = centers.float()[:, None] - state.float()[:, :, None]
    d = direction.float()[:, None, None]
    s = (r * d).sum(-1, keepdim=True)
    rho = torch.linalg.vector_norm(r - s * d, dim=-1, keepdim=True)
    return torch.cat((s, rho), -1)


def centered_attention(q, k, v):
    """Short self-attention with explicit FP32 probabilities and backward.

    Subtracting the mean key is a softmax-invariant row shift. It removes
    the large common logit offset without normalizing or clipping learned Q/K.
    Keeping probabilities avoids flash backward's reconstruction from FP32 LSE.
    Differentiating through centering also enforces the zero-sum key gradient.
    Callers select IEEE FP32 matmul rather than TF32 for this backend.
    """
    with torch.autocast(device_type=q.device.type, enabled=False):
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        qf, kf, vf = q.to(dtype), k.to(dtype), v.to(dtype)
        kf = kf - kf.mean(dim=-2, keepdim=True)
        logits = (qf * (q.shape[-1] ** -0.5)) @ kf.transpose(-1, -2)
        probabilities = logits.softmax(dim=-1)
        return (probabilities @ vf).to(q.dtype)


class Attention(nn.Module):
    def __init__(self, width, heads, backend='auto'):
        super().__init__()
        self.heads = heads
        self.backend = backend
        self.query = nn.Linear(width, width)
        self.key_value = nn.Linear(width, 2 * width)
        self.output = nn.Linear(width, width)

    def split(self, x):
        return x.unflatten(-1, (self.heads, -1)).transpose(1, 2)

    def project_kv(self, tokens):
        k, v = self.key_value(tokens).chunk(2, -1)
        return self.split(k), self.split(v)

    def forward(self, query, tokens=None, bias=None, kv=None):
        q = self.split(self.query(query))
        k, v = self.project_kv(tokens) if kv is None else kv
        if bias is not None:
            bias = bias.to(q.dtype)
        # PyTorch chooses a supported SDPA kernel; additive per-head bias stays differentiable.
        if self.backend == 'centered_fp32':
            if bias is not None:
                raise ValueError('centered_fp32 is for unbiased self-attention')
            value = centered_attention(q, k, v)
        else:
            value = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0)
        return self.output(value.transpose(1, 2).flatten(2))


class ConditionalBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        w = cfg.width
        branches = 2 if cfg.use_scene else 1
        self.norms = nn.ModuleList(nn.LayerNorm(w) for _ in range(branches+2))
        self.modulations = nn.ModuleList(nn.Sequential(nn.SiLU(), nn.Linear(w, 3*w)) for _ in range(branches+2))
        self.self_attention = Attention(w, cfg.heads, cfg.self_attention_backend)
        self.cross = nn.ModuleList(Attention(w, cfg.heads) for _ in range(branches))
        self.context_norms = nn.ModuleList(nn.LayerNorm(w) for _ in range(branches))
        self.biases = nn.ModuleList(nn.Sequential(nn.Linear(2, 64), nn.SiLU(),
                                                 nn.Linear(64, cfg.heads))
                                   for _ in range(0 if cfg.direction_adaln else branches))
        self.ffn = nn.Sequential(nn.Linear(w, 4*w), nn.GELU(), nn.Linear(4*w, w))
        self.checkpoint_direction_bias = False

    def modulate(self, x, condition, i):
        gamma, beta, gate = self.modulations[i](condition).chunk(3, -1)
        u = (1 + gamma[:, None]) * self.norms[i](x) + beta[:, None]
        return u, gate[:, None]

    def prepare_kv(self, tokens):
        return tuple(attn.project_kv(norm(g)) for attn, norm, g in
                     zip(self.cross, self.context_norms, tokens))

    def forward(self, x, condition, tokens, geometry, kv=None):
        u, gate = self.modulate(x, condition, 0)
        x = x + gate * self.self_attention(u, u)
        for i in range(len(self.cross)):
            u, gate = self.modulate(x, condition, i+1)
            bias = None
            if self.biases:
                value = (checkpoint(self.biases[i], geometry[i], use_reentrant=False, preserve_rng_state=False)
                         if self.checkpoint_direction_bias and self.training and torch.is_grad_enabled()
                         else self.biases[i](geometry[i]))
                bias = value.permute(0, 3, 1, 2)
            memory = self.context_norms[i](tokens[i]) if kv is None else None
            x = x + gate * self.cross[i](u, memory, bias, None if kv is None else kv[i])
        u, gate = self.modulate(x, condition, len(self.cross)+1)
        return x + gate * self.ffn(u)


@dataclass
class EncodedCondition:
    tokens: tuple
    centers: tuple
    gap_embedding: torch.Tensor | None
    direction: torch.Tensor
    kv: tuple | None = None
    direction_embedding: torch.Tensor | None = None


class KeypointModel(nn.Module):
    def __init__(self, cfg, method='fm', *, encoder_factory=GeometryEncoder):
        super().__init__()
        if method not in ('fm', 'flowmap'):
            raise ValueError('method must be fm or flowmap')
        self.cfg, self.method = cfg, method
        self.object_encoder = encoder_factory(cfg)
        self.scene_encoder = encoder_factory(cfg) if cfg.use_scene else None
        self.gap = nn.Sequential(nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, cfg.width)) if cfg.use_gap else None
        self.direction_embedding = (nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, cfg.width))
                                    if cfg.direction_adaln else None)
        self.time = nn.Sequential(nn.Linear(128 if method == 'fm' else 256, cfg.width),
                                  nn.SiLU(), nn.Linear(cfg.width, cfg.width))
        self.embed = nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, cfg.width))
        self.ids = nn.Embedding(cfg.keypoints, cfg.width)
        self.blocks = nn.ModuleList(ConditionalBlock(cfg) for _ in range(cfg.depth))
        self.head = nn.Sequential(nn.LayerNorm(cfg.width), nn.Linear(cfg.width, 3))

    def encode(self, condition, *, cache_kv=False):
        if cache_kv and (self.training or torch.is_grad_enabled()):
            raise ValueError('K/V caching is inference-only; call eval() under no_grad')
        missing = set(self.cfg.condition_keys) - condition.keys()
        if missing:
            raise ValueError(f'missing model conditions: {sorted(missing)}')
        obj, direction = condition['object_points'], condition['grasp_direction_cmd']
        b = len(obj)
        if obj.shape != (b, 1024, 3) or direction.shape != (b, 3):
            raise ValueError('invalid object/direction dimensions')
        if not all(torch.isfinite(condition[k]).all() for k in self.cfg.condition_keys):
            raise ValueError('conditions must be finite')
        if self.cfg.use_scene:
            scene, mask = condition['scene_points'], condition['scene_mask']
            if scene.shape != (b, 2048, 3) or mask.shape != (b, 2048):
                raise ValueError('v1 requires scene[2048,3], mask[2048]')
            if not ((mask == 0) | (mask == 1)).all() or not (mask.sum(-1) == 512).all():
                raise ValueError('scene_mask must contain exactly 512 target points')
        if self.cfg.use_gap:
            gap = condition['palm_gap']
            if gap.shape != (b, 1) or (gap < 0).any():
                raise ValueError('palm_gap must be nonnegative with shape [B,1]')
        norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        if (norm <= 1e-8).any():
            raise ValueError('grasp direction must be nonzero')
        direction = direction / norm
        object_options = {'indices': condition['object_fps_indices']} if 'object_fps_indices' in condition else {}
        go, co = self.object_encoder(obj, torch.ones_like(obj[..., 0]), **object_options)
        tokens, centers = (go,), (co,)
        if self.cfg.use_scene:
            scene_options = {'indices': condition['scene_fps_indices']} if 'scene_fps_indices' in condition else {}
            gs, cs = self.scene_encoder(scene, mask, **scene_options)
            tokens, centers = (go, gs), (co, cs)
        kv = tuple(block.prepare_kv(tokens) for block in self.blocks) if cache_kv else None
        gap_embedding = self.gap(gap / self.cfg.hand_reference_length_m) if self.cfg.use_gap else None
        direction_embedding = self.direction_embedding(direction) if self.cfg.direction_adaln else None
        return EncodedCondition(tokens, centers, gap_embedding, direction, kv, direction_embedding)

    def forward(self, state, time_start, condition, time_end=None):
        if state.shape != (len(state), self.cfg.keypoints, 3) or time_start.shape != (len(state),):
            raise ValueError('invalid keypoint state or time shape')
        if self.method == 'fm' and time_end is not None:
            raise ValueError('FM takes one time; FlowMap takes a time pair')
        if self.method == 'flowmap' and (time_end is None or time_end.shape != time_start.shape):
            raise ValueError('FlowMap requires matching start/end times')
        if not torch.isfinite(time_start).all() or ((time_start < 0) | (time_start > 1)).any():
            raise ValueError('start time must be in [0,1]')
        if time_end is not None and (not torch.isfinite(time_end).all() or ((time_end < time_start) | (time_end > 1)).any()):
            raise ValueError('FlowMap requires 0 <= start <= end <= 1')
        times = time_embedding(time_start)
        if self.method == 'flowmap':
            times = torch.cat((times, time_embedding(time_end)), -1)
        global_condition = self.time(times)
        if condition.gap_embedding is not None:
            global_condition = condition.gap_embedding + global_condition
        if condition.direction_embedding is not None:
            global_condition = global_condition + condition.direction_embedding
        x = self.embed(state) + self.ids.weight[None]
        geometry = (() if self.cfg.direction_adaln else
                    tuple(direction_geometry(state, c, condition.direction) for c in condition.centers))
        for i, block in enumerate(self.blocks):
            x = block(x, global_condition, condition.tokens, geometry,
                      None if condition.kv is None else condition.kv[i])
        value = self.head(x)
        if self.method == 'flowmap':
            # Accumulate coordinates in FP32 even under BF16 autocast.
            value = state.float() + (time_end - time_start)[:, None, None] * value.float()
        return value
