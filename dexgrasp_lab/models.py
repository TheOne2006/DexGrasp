"""Two controlled baselines, not replicas of the published architectures."""
import math
import torch
from torch import nn

ARCHITECTURES = ("pointnet_mlp", "pointnet_transformer")


class GraspModel(nn.Module):
    def __init__(self, architecture="pointnet_mlp", width=128, depth=2):
        super().__init__()
        if architecture not in ARCHITECTURES or width < 8 or width % 8 or depth < 1:
            raise ValueError("unknown architecture or invalid width/depth")
        self.architecture, self.width = architecture, width
        self.encoder = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.time = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        if architecture == "pointnet_mlp":
            layers = [nn.Linear(252 + 2 * width, width), nn.SiLU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(width, width), nn.SiLU()]
            self.decoder = nn.Sequential(*layers, nn.Linear(width, 252))
        else:
            self.input = nn.Linear(3, width)
            self.position = nn.Parameter(torch.randn(1, 84, width) * 0.02)
            layer = nn.TransformerEncoderLayer(width, 8, 4 * width, dropout=0.0,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.decoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
            self.output = nn.Linear(width, 3)

    def encode(self, point_cloud):
        return self.encoder(point_cloud).amax(dim=1)

    def forward(self, x, t, context):
        freq = torch.exp(-math.log(10000) * torch.arange(self.width // 2, device=x.device)
                         / (self.width // 2))
        phase = t[:, None] * freq[None] * 1000
        time = self.time(torch.cat([phase.sin(), phase.cos()], dim=-1))
        if self.architecture == "pointnet_mlp":
            return self.decoder(torch.cat([x.flatten(1), context, time], dim=-1)).reshape_as(x)
        tokens = self.input(x.reshape(-1, 84, 3)) + self.position
        tokens = tokens + context[:, None] + time[:, None]
        return self.output(self.decoder(tokens)).reshape_as(x)
