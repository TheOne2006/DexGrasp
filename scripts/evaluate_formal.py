"""Prepare a frozen selection, then evaluate existing formal packed checkpoints."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from dexgrasp.test.formal import prepare, run, compare
    import torch
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--split', choices=['train','test'], default='test')
    p.add_argument('--conditions', type=int, default=4)
    p.add_argument('--seed', type=int, default=7301)
    p.add_argument('--max-objects', type=int)
    p = sub.add_parser('run')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--selection', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--candidates', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=8401)
    p.add_argument('--steps', type=int, default=20)
    p = sub.add_parser('compare')
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--output', required=True)
    args = vars(parser.parse_args())
    command = args.pop('command')
    {'prepare':prepare, 'run':run, 'compare':compare}[command](**args)


if __name__ == '__main__':
    main()
