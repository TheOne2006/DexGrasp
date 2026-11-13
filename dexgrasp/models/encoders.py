"""MinkUNet point-to-voxel-to-point encoding; no hidden dense fallback."""
import torch
from torch import nn


def selected_point_features(output, field, indices, points_per_cloud):
    """Compose the inverse voxel map with FPS before gathering feature channels."""
    rows = torch.arange(len(indices), device=indices.device)[:, None]
    point_ids = rows*points_per_cloud + indices
    inverse = field.inverse_mapping(output.coordinate_map_key).long()
    return output.F[inverse[point_ids]]


@torch.no_grad()
def farthest_point_indices(points, count):
    """Deterministic FPS, starting farthest from the centroid; O(B*N*count)."""
    if points.ndim != 3 or points.shape[-1] != 3 or not 0 < count <= points.shape[1]:
        raise ValueError('invalid FPS input/count')
    b, n, _ = points.shape
    xyz = points.float()
    chosen = torch.empty((b, count), dtype=torch.long, device=points.device)
    distance = torch.full((b, n), torch.inf, device=points.device)
    farthest = (xyz - xyz.mean(1, keepdim=True)).square().sum(-1).argmax(-1)
    rows = torch.arange(b, device=points.device)
    selected = torch.zeros((b, n), dtype=torch.bool, device=points.device)
    for i in range(count):
        chosen[:, i] = farthest
        selected[rows, farthest] = True
        delta = xyz - xyz[rows, farthest, None]
        distance = torch.minimum(distance, delta.square().sum(-1))
        farthest = distance.masked_fill(selected, -1).argmax(-1)
    return chosen


class GeometryEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        try:
            import MinkowskiEngine as ME
            from ..vendor.mink_unet import MinkUNet14
        except ImportError as exc:
            raise ImportError('MinkowskiEngine is required for the v1 geometry encoders; '
                              'no alternative backbone is substituted.') from exc
        self.ME = ME
        self.backbone = MinkUNet14(in_channels=1, out_channels=cfg.backbone_channels)
        self.voxel_size = cfg.voxel_size_m / cfg.coord_scale_m
        self.tokens = cfg.tokens
        self.fps_backend = 'torch'
        self.select_tokens_before_slice = False
        self.project = nn.Linear(cfg.backbone_channels, cfg.width)
        self.position = nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, cfg.width))

    def forward(self, points, mask, *, indices=None):
        b, n, _ = points.shape
        if points.is_cuda and not self.ME.is_cuda_available():
            raise RuntimeError('CUDA point clouds require a CUDA-enabled MinkowskiEngine build')
        # Keep sparse operations in FP32; Transformer may use BF16 independently.
        with torch.autocast(device_type=points.device.type, enabled=False):
            batch = torch.arange(b, device=points.device).repeat_interleave(n)[:, None]
            coords = torch.cat((batch, points.reshape(-1, 3).float() / self.voxel_size), -1)
            field = self.ME.TensorField(
                features=mask.reshape(-1, 1).float(), coordinates=coords.float(),
                quantization_mode=self.ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                device=points.device,
            )
            # Averaging preserves mixed target/background voxels instead of choosing a random label.
            output = self.backbone(field.sparse())
        if indices is None:
            if self.fps_backend == 'triton':
                from .fps_triton import farthest_point_indices_triton
                indices = farthest_point_indices_triton(points, self.tokens)
            else:
                indices = farthest_point_indices(points, self.tokens)
        elif indices.shape != (b, self.tokens) or indices.dtype != torch.long or indices.device != points.device:
            raise ValueError('invalid cached FPS indices')
        rows = torch.arange(b, device=points.device)[:, None]
        centers = points[rows, indices]
        features = (selected_point_features(output, field, indices, n) if self.select_tokens_before_slice
                    else output.slice(field).F.reshape(b, n, -1)[rows, indices])
        tokens = self.project(features) + self.position(centers)
        return tokens, centers
