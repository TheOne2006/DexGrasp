"""Find a synthetic DDP microbatch boundary, restarting all ranks after each trial."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--ranks', type=int, default=4)
    parser.add_argument('--start', required=True, type=int)
    parser.add_argument('--stride', type=int, default=32, help='Initial bracketing step')
    parser.add_argument('--resolution', type=int, default=1)
    args, options = parser.parse_known_args()
    if min(args.ranks, args.start, args.stride, args.resolution) < 1 or args.start % args.resolution:
        parser.error('positive ranks, start, stride, resolution and an aligned starting batch are required')
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    trials = []

    def fits(batch):
        report = output/f'b{batch}.json'
        log = output/f'b{batch}.log'
        cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               f'--nproc-per-node={args.ranks}', str(root/'scripts/benchmark_fake.py'),
               *options, '--batch', str(batch), '--output', str(report)]
        with log.open('x') as stream:
            proc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
        if proc.returncode:
            if not re.search(r'out of memory|cudaErrorMemoryAllocation|CUBLAS_STATUS_ALLOC_FAILED',
                             log.read_text(), re.I):
                raise RuntimeError(f'non-capacity failure, inspect {log}')
            trials.append(dict(batch=batch, outcome='OOM'))
            return False
        result = json.loads(report.read_text())
        trials.append(dict(batch=batch, outcome='pass', samples_per_second=result['samples_per_second']))
        return True

    q = args.resolution
    delta = max(1, args.stride//q)*q
    if fits(args.start):
        low = args.start
        high = low + delta
        while fits(high):
            low, delta = high, delta*2
            high = low + delta
    else:
        high = args.start
        low = max(0, high-delta)
        while low and not fits(low):
            high, delta = low, delta*2
            low = max(0, high-delta)
    while high-low > q:
        batch = low + ((high-low)//(2*q))*q
        if fits(batch):
            low = batch
        else:
            high = batch
    summary = dict(max_pass_per_gpu=low, next_oom_per_gpu=high, ranks=args.ranks,
                   resolution=q, trials=trials)
    (output/'capacity.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
