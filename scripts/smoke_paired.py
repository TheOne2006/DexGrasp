"""Exercise full-size DDP, checkpoint resume and inference on paired synthetic HDF5."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from test_three_stage import paired_shard


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--name', required=True, help='New directory name for this synthetic run')
    p.add_argument('--frozen-grasp', type=Path, help='Initialize the two-head model from this FM checkpoint')
    p.add_argument('--hand-spec', type=Path, help='Use the same hand calibration as the parent checkpoint')
    p.add_argument('--ranks', type=int, default=4)
    a = p.parse_args()
    if Path(a.name).name != a.name or a.name in ('', '.', '..') or a.ranks < 1:
        p.error('use a single new directory name and a positive rank count')
    workspace = ROOT.parent
    for parent in ('data', 'checkpoints'):
        link = workspace / parent
        if not link.is_symlink() or not link.is_dir():
            raise RuntimeError(f'missing team storage link {link}')
    data, out, weights = (workspace / folder / 'synthetic_smoke' / a.name
                          for folder in ('data', 'outputs', 'checkpoints'))
    if any(d.exists() for d in (data, out, weights)):
        raise FileExistsError('choose a new smoke run name')
    data.mkdir(parents=True)
    out.mkdir(parents=True)
    paired_shard(data, a.hand_spec)
    cfg = dict(model_config=str(a.model.resolve()), dataset_format='data_v2', method='fm',
               dataset_root=str(data), hand_spec=str(data/'hand/hand_spec.json'),
               run_dir=str(out/'train'), checkpoint_dir=str(weights), device='cuda:0',
               precision='bfloat16', batch_size=2, accumulation=2, cuda_prefetch=True,
               num_workers=1, max_steps=2, save_every=1, optimizer='adamw',
               lr=.0002, weight_decay=.01, lr_schedule='constant', seed=42,
               fps_backend='triton', select_tokens_before_slice=True,
               compile_transformer=True, checkpoint_direction_bias=True)
    if a.frozen_grasp:
        cfg.update(initialization='frozen_grasp', initial_checkpoint=str(a.frozen_grasp.resolve()))
    config = out / 'recipe.json'
    config.write_text(json.dumps(cfg, indent=2))
    trainer = ['-m', 'torch.distributed.run', '--standalone', f'--nproc-per-node={a.ranks}',
               'scripts/train_fm_ddp.py', '--config', str(config)]
    def execute(args, log):
        with (out/log).open('x') as stream:
            subprocess.run([sys.executable, *args], cwd=ROOT, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
    execute([*trainer, '--stop-after', '1'], 'step1.log')
    execute([*trainer, '--resume', str(weights/'last.pt'), '--stop-after', '2'], 'resume.log')
    execute(['-m','dexgrasp','sample','--checkpoint',str(weights/'last.pt'),'--dataset-root',str(data),
             '--output',str(out/'samples.h5'),'--device','cuda:0','--steps','2','--candidates','1'], 'sample.log')
    execute(['-m','dexgrasp','decode','--samples',str(out/'samples.h5'),
             '--output',str(out/'decoded.jsonl')], 'decode.log')
    execute(['-m','dexgrasp','test','--decoded',str(out/'decoded.jsonl')], 'evaluation.json')
    metrics = [json.loads(line) for line in (out/'train/metrics.jsonl').read_text().splitlines()]
    if [r['step'] for r in metrics] != [1,2] or json.loads((out/'evaluation.json').read_text())['candidates'] != 2:
        raise RuntimeError('incomplete resumed updates or candidate count')
    print(out)


if __name__ == '__main__':
    main()
