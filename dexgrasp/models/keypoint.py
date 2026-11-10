"""Shared v1 conditional Transformer; independent FM and FlowMap parameters."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
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


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
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
        value = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0)
        return self.output(value.transpose(1, 2).flatten(2))


class ConditionalBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        w = cfg.width
        self.norms = nn.ModuleList(nn.LayerNorm(w) for _ in range(4))
        self.modulations = nn.ModuleList(nn.Sequential(nn.SiLU(), nn.Linear(w, 3*w)) for _ in range(4))
        self.self_attention = Attention(w, cfg.heads)
        self.cross = nn.ModuleList(Attention(w, cfg.heads) for _ in range(2))
        self.context_norms = nn.ModuleList(nn.LayerNorm(w) for _ in range(2))
        self.biases = nn.ModuleList(nn.Sequential(nn.Linear(2, 64), nn.SiLU(),
                                                 nn.Linear(64, cfg.heads)) for _ in range(2))
        self.ffn = nn.Sequential(nn.Linear(w, 4*w), nn.GELU(), nn.Linear(4*w, w))

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
        for i in range(2):
            u, gate = self.modulate(x, condition, i+1)
            bias = self.biases[i](geometry[i]).permute(0, 3, 1, 2)
            memory = self.context_norms[i](tokens[i]) if kv is None else None
            x = x + gate * self.cross[i](u, memory, bias, None if kv is None else kv[i])
        u, gate = self.modulate(x, condition, 3)
        return x + gate * self.ffn(u)


@dataclass
class EncodedCondition:
    tokens: tuple
    centers: tuple
    gap_embedding: torch.Tensor
    direction: torch.Tensor
    kv: tuple | None = None


class KeypointModel(nn.Module):
    def __init__(self, cfg, method='fm', *, encoder_factory=GeometryEncoder):
        super().__init__()
        if method not in ('fm', 'flowmap'):
            raise ValueError('method must be fm or flowmap')
        self.cfg, self.method = cfg, method
        self.object_encoder = encoder_factory(cfg)
        self.scene_encoder = encoder_factory(cfg)
        self.gap = nn.Sequential(nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, cfg.width))
        self.time = nn.Sequential(nn.Linear(128 if method == 'fm' else 256, cfg.width),
                                  nn.SiLU(), nn.Linear(cfg.width, cfg.width))
        self.embed = nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, cfg.width))
        self.ids = nn.Embedding(cfg.keypoints, cfg.width)
        self.blocks = nn.ModuleList(ConditionalBlock(cfg) for _ in range(cfg.depth))
        self.head = nn.Sequential(nn.LayerNorm(cfg.width), nn.Linear(cfg.width, 3))

    def encode(self, condition, *, cache_kv=False):
        if cache_kv and (self.training or torch.is_grad_enabled()):
            raise ValueError('K/V caching is inference-only; call eval() under no_grad')
        obj, scene, mask = (condition[k] for k in ('object_points', 'scene_points', 'scene_mask'))
        gap, direction = condition['palm_gap'], condition['grasp_direction_cmd']
        b = len(obj)
        if obj.shape != (b, 1024, 3) or scene.shape != (b, 2048, 3) or mask.shape != (b, 2048):
            raise ValueError('v1 requires object[1024,3], scene[2048,3], mask[2048]')
        if gap.shape != (b, 1) or direction.shape != (b, 3):
            raise ValueError('invalid gap/direction dimensions')
        if not all(torch.isfinite(x).all() for x in (obj, scene, gap, direction)) or (gap < 0).any():
            raise ValueError('conditions must be finite and palm_gap nonnegative')
        if not ((mask == 0) | (mask == 1)).all() or not (mask.sum(-1) == 512).all():
            raise ValueError('scene_mask must contain exactly 512 target points')
        norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        if (norm <= 1e-8).any():
            raise ValueError('grasp direction must be nonzero')
        direction = direction / norm
        object_options = {'indices': condition['object_fps_indices']} if 'object_fps_indices' in condition else {}
        scene_options = {'indices': condition['scene_fps_indices']} if 'scene_fps_indices' in condition else {}
        go, co = self.object_encoder(obj, torch.ones_like(obj[..., 0]), **object_options)
        gs, cs = self.scene_encoder(scene, mask, **scene_options)
        tokens = (go, gs)
        kv = tuple(block.prepare_kv(tokens) for block in self.blocks) if cache_kv else None
        return EncodedCondition(tokens, (co, cs), self.gap(gap / self.cfg.hand_reference_length_m),
                                direction, kv)

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
        global_condition = condition.gap_embedding + self.time(times)
        x = self.embed(state) + self.ids.weight[None]
        geometry = tuple(direction_geometry(state, c, condition.direction) for c in condition.centers)
        for i, block in enumerate(self.blocks):
            x = block(x, global_condition, condition.tokens, geometry,
                      None if condition.kv is None else condition.kv[i])
        value = self.head(x)
        if self.method == 'flowmap':
            # Accumulate coordinates in FP32 even under BF16 autocast.
            value = state.float() + (time_end - time_start)[:, None, None] * value.float()
        return value
