"""Evaluate a frozen checkpoint on fresh noise; paired errors are not grasp success."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dexgrasp.data.dataset import GraspDataset, audit_dataset, sha256
from dexgrasp.train.runner import load_model, move_batch
from dexgrasp.train.objectives import sample_keypoints


@torch.no_grad()
def evaluate(checkpoint, output, device='cuda:0', seeds=32, batch_size=8):
    if output.exists():
        raise FileExistsError(output)
    model, state = load_model(checkpoint, device)
    cfg = state['config']
    audit = audit_dataset(cfg['dataset_root'], model.cfg.coord_scale_m)
    # The runner adds the URDF hash to data_identity separately.
    assert all(state['data_identity'][k] == v for k, v in audit.items())
    summary = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha256(checkpoint),
                   method=model.method, optimizer_steps=state['step'], seed_start=40000,
                   seeds=seeds, batch_size=batch_size, sampling_steps=cfg['sample_steps'],
                   ranking='none; every sample counted', physical_success_rate=None,
                   scope='kinematic diagnostic; no claim of grasp or generalization success', splits={})
    arrays = {}
    for split in ('train', 'val', 'test'):
        ds = GraspDataset(cfg['dataset_root'], split, model.cfg.coord_scale_m)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        predictions, targets, centers = [], [], []
        for seed in range(40000, 40000+seeds):
            torch.manual_seed(seed)
            chunks = []
            for batch in loader:
                batch = move_batch(batch, device)
                chunks.append(sample_keypoints(model, batch['condition'], steps=cfg['sample_steps']).cpu())
                if seed == 40000:
                    targets.append(batch['keypoints_gt'].cpu())
                    centers.append(batch['metadata']['center_m'].cpu())
            predictions.append(torch.cat(chunks))
        predicted = torch.stack(predictions)
        gt, center = torch.cat(targets), torch.cat(centers)
        error = (predicted-gt[None]).norm(dim=-1)*model.cfg.coord_scale_m*1000
        per_sample = error.mean((0, 2))
        summary['splits'][split] = dict(samples=len(ds), mpjpe_mm=float(error.mean()),
                                        p95_keypoint_mm=float(torch.quantile(error.flatten(), .95)),
                                        max_sample_mpjpe_mm=float(per_sample.max()),
                                        per_sample_mpjpe_mm=per_sample.tolist(),
                                        condition_mean_pose_mpjpe_mm=float((gt-gt.mean(0)).norm(dim=-1).mean()*model.cfg.coord_scale_m*1000))
        arrays[split+'_prediction_m'] = (predicted*model.cfg.coord_scale_m+center[None, :, None]).cpu().numpy()
        arrays[split+'_target_m'] = (gt*model.cfg.coord_scale_m+center[:, None]).cpu().numpy()
        ds.close()
    summary['overfit_target_mpjpe_mm'] = 5.
    summary['overfit_target_met'] = summary['splits']['train']['mpjpe_mm'] < 5.
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix('.npz'), **arrays)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seeds', type=int, default=32)
    p.add_argument('--batch-size', type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(4)
    evaluate(args.checkpoint, args.output, args.device, args.seeds, args.batch_size)
