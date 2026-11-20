"""Run the public diversity CLI inside the existing GPU ownership/resource guard.

Guard config: worker=diversity_worker.py, diversity_args=[CLI arguments].
The guard supplies CUDA_VISIBLE_DEVICES; this worker uses its logical cuda:0.
"""
import json
import os
from pathlib import Path
import sys


def main():
    config_path, output = map(Path, sys.argv[1:])
    config = json.loads(config_path.read_text())
    identity = json.loads(Path(os.environ['DEXONOMY_PID_RECEIPT']).read_text())
    if identity['status'] != 'PASS' or identity['local_pid'] != os.getpid():
        raise RuntimeError('GPU guard ownership is required')
    args = config['diversity_args']
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise ValueError('diversity_args must be a list of CLI argument strings')
    output.mkdir(parents=True, exist_ok=False)
    from dexonomy.diversity.__main__ import main as evaluate
    evaluate([*args, '--device', 'cuda:0', '--output', str(output / 'diversity.json')])
    if not (output / 'diversity.json').is_file():
        raise RuntimeError('No diversity result produced')
    print('DIVERSITY COMPLETE', str(output / 'diversity.json'), flush=True)


if __name__ == '__main__':
    main()
