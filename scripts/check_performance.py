"""Synthetic forward/backward and cache timing, never an optimizer update or grasp experiment."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dexgrasp.config import ModelConfig
from dexgrasp.models.encoders import GeometryEncoder
from dexgrasp.models.keypoint import KeypointModel
from dexgrasp.train.objectives import fm_loss, sample_keypoints, flowmap_loss
from dexgrasp.train.runner import source_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True)
    parser.add_argument('--transformer-only', action='store_true', help='Use the test-only geometry double; excludes sparse backbone')
    parser.add_argument('--small', action='store_true', help='Use reduced Transformer dimensions for CPU checks')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError('repeats must be positive')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = torch.device(args.device)
    factory = GeometryEncoder
    if args.transformer_only:
        sys.path.insert(0, str(ROOT/'tests'))
        from helpers import FixtureEncoder
        factory = FixtureEncoder
    cfg = ModelConfig(hand_reference_length_m=.1,
                      **(dict(width=32, depth=2, heads=4, tokens=8, backbone_channels=16) if args.small else {}))
    condition = dict(object_points=torch.randn(2, 1024, 3, device=device)*.15,
                     scene_points=torch.randn(2, 2048, 3, device=device)*.2,
                     scene_mask=torch.cat((torch.ones(2, 512), torch.zeros(2, 1536)), 1).to(device),
                     palm_gap=torch.full((2, 1), .02, device=device),
                     grasp_direction_cmd=torch.tensor([[0., 0., 1.]], device=device).repeat(2, 1))
    batch = dict(condition=condition, keypoints_gt=torch.randn(2, 21, 3, device=device))
    teacher = KeypointModel(cfg, 'fm', encoder_factory=factory).to(device)
    loss = fm_loss(teacher, batch)
    loss.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in teacher.parameters()):
        raise AssertionError('nonfinite or disconnected FM gradient')
    teacher.zero_grad(set_to_none=True)
    teacher.requires_grad_(False)
    teacher.eval()
    student = KeypointModel(cfg, 'flowmap', encoder_factory=factory).to(device)
    map_loss = flowmap_loss(student, teacher, batch, steps=2)
    map_loss.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters()):
        raise AssertionError('nonfinite or disconnected FlowMap gradient')
    student.zero_grad(set_to_none=True)
    noise = torch.randn(2, 21, 3, device=device)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def measure(model, steps, cache):
        for _ in range(2):
            sample_keypoints(model, condition, steps=steps, noise=noise, cache_kv=cache)
        times, value = [], None
        for _ in range(args.repeats):
            sync()
            started = time.perf_counter()
            value = sample_keypoints(model, condition, steps=steps, noise=noise, cache_kv=cache)
            sync()
            times.append(time.perf_counter()-started)
        return statistics.median(times), value

    uncached_seconds, uncached = measure(teacher, 10, False)
    cached_seconds, cached = measure(teacher, 10, True)
    torch.testing.assert_close(cached, uncached, atol=2e-5, rtol=2e-5)
    map_seconds, _ = measure(student, 1, True)
    report = dict(scope='synthetic software performance, not grasp quality',
                  geometry_backend='test double; sparse backbone excluded' if args.transformer_only else 'MinkUNet14',
                  model_config=asdict(cfg), device=str(device), torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                  fm_parameters=sum(p.numel() for p in teacher.parameters()),
                  flowmap_parameters=sum(p.numel() for p in student.parameters()),
                  batch_size=2, repeats=args.repeats, fm_loss=float(loss.detach()),
                  flowmap_loss=float(map_loss.detach()), gradients_finite=True, optimizer_updates=0,
                  fm10_uncached_median_seconds=uncached_seconds, fm10_cached_median_seconds=cached_seconds,
                  kv_cache_speedup=uncached_seconds/cached_seconds,
                  cache_max_absolute_error=float((cached-uncached).abs().max()),
                  flowmap1_median_seconds=map_seconds,
                  excludes='SVD/IK, collision checks, physics and planning', source=source_state())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k != 'source'}, indent=2))


if __name__ == '__main__':
    main()
