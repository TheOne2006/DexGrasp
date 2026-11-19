"""Predict G7 endpoints around archived A/B/C/D grasps on the frozen evaluation draws."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from dexgrasp.models.checkpoint import load_model
from dexgrasp.models.frozen_residual import integrate_residual
from dexgrasp.train.execution import configure_execution
from dexgrasp.test.frozen import digest, save

BASE = REPO.parent
FROZEN = BASE / 'outputs/final_full_abcd_20261117'
OLD = BASE / 'outputs/gsr_osr_200k'


def source(suite):
    if suite == 'paired':
        return FROZEN, 'test'
    if suite == 'official':
        return OLD / suite, suite
    if suite in ('noise', 'random'):
        return OLD / (suite + '_inputs'), suite
    return OLD / 'noise_scale' / suite, 'noise'


@torch.inference_mode()
def endpoints(model, points, directions, grasp, noise):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        context = model.encode(dict(object_points=points, grasp_direction_cmd=directions))
        hidden = model.grasp.features(grasp, grasp.new_ones(len(grasp)), context)
        features = (grasp, hidden, context.tokens[0], 20, True)
        return torch.stack([
            grasp + integrate_residual(flow, noise[:, i], *features)
            for i, flow in enumerate((model.pregrasp, model.squeeze))
        ], dim=1)


def generate(args):
    torch.set_num_threads(1)
    device = torch.device('cuda:0')
    model, state = load_model(args.checkpoint, device)
    if model.cfg.family != 'frozen_grasp_residual' or state['step'] != args.step:
        raise ValueError('Wrong G7 checkpoint')
    configure_execution(model, dict(state['config'], compile_transformer=False))
    del state
    plan = json.loads((FROZEN / 'plan.json').read_text())
    data = Path(plan['root'])
    cloud = np.load(data / 'object_points.npy', mmap_mode='r')
    obs = np.load(data / 'observation_index.npy', mmap_mode='r')
    official = np.load(OLD / 'official/points.npy', mmap_mode='r')
    paired = np.load(OLD / 'internal_stages/paired.npy')
    scale = model.cfg.coord_scale_m
    checkpoint_hash = digest(args.checkpoint)
    def gpu(x):
        return torch.as_tensor(np.asarray(x).copy(), device=device, dtype=torch.float32)
    for suite in args.suites:
        root, mode = source(suite)
        for group in args.groups:
            rank = args.rank
            src = root / group / f'rank{rank}' / mode
            dst = args.output / suite / group / f'rank{rank}'
            dst.mkdir(parents=True, exist_ok=True)
            with np.load(src / 'index.npz') as z:
                index = dict(z)
            keys = np.load(src / 'keypoints.npy', mmap_mode='r')
            rows = paired[paired % 2 == rank] // 2 if suite == 'paired' else np.arange(len(keys))
            if args.limit:
                rows = rows[:args.limit]
            index = {k: v[rows] for k, v in index.items()}
            ids = index['index']
            np.testing.assert_array_equal(ids, 2 * rows + rank)
            receipt = dict(checkpoint=checkpoint_hash, step=args.step, suite=suite, group=group,
                           index_sha256=digest(src / 'index.npz'), anchor_sha256=digest(src / 'keypoints.npy'),
                           count=len(rows), code=digest(__file__), seed=920431, model_steps=20)
            done = dst / 'generation.json'
            if done.exists():
                prior = json.loads(done.read_text())
                if not all(prior[k] == v for k, v in receipt.items()):
                    raise ValueError(f'Changed generation inputs in {dst}')
                continue
            np.savez(dst / 'index.npz', **index)
            result = np.lib.format.open_memmap(dst / 'endpoints.npy', mode='w+', dtype='f4',
                                               shape=(len(rows), 2, 21, 3))
            elapsed = 0.
            if suite == '1.0' and args.limit is None:
                # The unit-scale arm is the exact subset of the 32-draw noise experiment.
                unit = args.output / 'noise' / group / f'rank{rank}'
                unit_receipt = json.loads((unit / 'generation.json').read_text())
                if unit_receipt['checkpoint'] != checkpoint_hash:
                    raise ValueError('Unit-scale reference checkpoint changed')
                original = index['original_index']
                with np.load(unit / 'index.npz') as full:
                    np.testing.assert_array_equal(full['index'][original // 2], original)
                    np.testing.assert_array_equal(full['packed'][original // 2], index['packed'])
                result[:] = np.load(unit / 'endpoints.npy', mmap_mode='r')[original // 2]
                result.flush()
                save(done, dict(receipt, synchronized_seconds=0., subset_of=digest(unit / 'generation.json'),
                                finite=int(np.isfinite(result).all(axis=(1,2,3)).sum())))
                continue
            for start in range(0, len(rows), args.batch):
                sl = slice(start, start + args.batch)
                batch = ids[sl]
                raw_grasp = gpu(keys[rows[sl]])
                if suite == 'official':
                    points = gpu(official[batch // 10])
                    center = points.double().mean(1).float()[:, None]
                    points = (points - center) / scale
                else:
                    points = gpu(cloud[obs[index['packed'][sl]]]) * (0.2 / scale)
                    center = torch.zeros_like(raw_grasp[:, :1])
                grasp = (raw_grasp - center) / scale
                if suite == 'random':
                    seeds = index['packed'][sl]
                else:
                    seeds = index.get('original_index', ids)[sl]
                noise = torch.stack([torch.randn(2, 21, 3, generator=torch.Generator().manual_seed(
                    920431 + int(i))) for i in seeds]).to(device)
                directions = gpu(index['directions'][sl])
                torch.cuda.synchronize()
                tick = time.perf_counter()
                predicted = endpoints(model, points, directions, grasp, noise) * scale + center[:, None]
                torch.cuda.synchronize()
                elapsed += time.perf_counter() - tick
                result[sl] = predicted.float().cpu().numpy()
            result.flush()
            save(done, dict(receipt, synchronized_seconds=elapsed,
                            finite=int(np.isfinite(result).all(axis=(1,2,3)).sum())))
            print(suite, group, rank, len(rows), elapsed, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rank', type=int, choices=(0,1), required=True)
    p.add_argument('--groups', default='ABCD')
    p.add_argument('--suites', nargs='+', choices=('paired','random','noise','official','0.5','1.0','1.5'),
                   default=['paired','random','noise','0.5','1.0','1.5','official'])
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--limit', type=int)
    generate(p.parse_args())
