"""Find a synthetic DDP microbatch boundary, restarting all ranks after each trial."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


def standard_batches():
    power, five_times_power = 1, 5
    while True:
        if power < five_times_power:
            yield power
            power *= 2
        else:
            yield five_times_power
            five_times_power *= 2


def standard_boundary(start, fits, *, fallback_1024=True):
    def retry_below(high, low):
        if fallback_1024 and high == 1024:
            for fallback in (1000, 960):
                if fits(fallback):
                    return fallback, high
                high = fallback
        return low, high

    levels = standard_batches()
    seen = []
    for batch in levels:
        seen.append(batch)
        if batch >= start:
            break
    index = min(range(len(seen)), key=lambda i: abs(seen[i] - start))
    if fits(seen[index]):
        low = seen[index]
        while True:
            index += 1
            if index == len(seen):
                seen.append(next(levels))
            if not fits(seen[index]):
                return retry_below(seen[index], low)
            low = seen[index]
    low, high = retry_below(seen[index], 0)
    if low:
        return low, high
    for lower in reversed(seen[:index]):
        if fits(lower):
            return lower, high
        low, high = retry_below(lower, 0)
        if low:
            return low, high
    return 0, high


def exact_boundary(start, stride, resolution, fits):
    delta = max(1, stride//resolution)*resolution
    if fits(start):
        low = start
        high = low + delta
        while fits(high):
            low, delta = high, delta*2
            high = low + delta
    else:
        high = start
        low = max(0, high-delta)
        while low and not fits(low):
            high, delta = low, delta*2
            low = max(0, high-delta)
    while high-low > resolution:
        batch = low + ((high-low)//(2*resolution))*resolution
        if fits(batch):
            low = batch
        else:
            high = batch
    return low, high


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--ranks', type=int, default=4)
    parser.add_argument('--start', required=True, type=int)
    parser.add_argument('--exact', action='store_true', help='Search an integer grid instead of 2^n and 5*2^n')
    parser.add_argument('--stride', type=int, help='Use the exact grid with this initial bracketing step')
    parser.add_argument('--resolution', type=int, help='Use the exact grid with this batch increment')
    args, options = parser.parse_known_args()
    exact = args.exact or args.stride is not None or args.resolution is not None
    stride = 32 if args.stride is None else args.stride
    resolution = 1 if args.resolution is None else args.resolution
    if min(args.ranks, args.start, stride, resolution) < 1 or args.start % resolution:
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

    if exact:
        low, high = exact_boundary(args.start, stride, resolution, fits)
    else:
        low, high = standard_boundary(args.start, fits)
    summary = dict(max_pass_per_gpu=low, next_oom_per_gpu=high, ranks=args.ranks,
                   max_pass_global=low*args.ranks, next_oom_global=high*args.ranks,
                   grid='exact' if exact else '2^n or 5*2^n, with 1000/960 fallback below 1024',
                   resolution=resolution if exact else None, trials=trials)
    (output/'capacity.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
