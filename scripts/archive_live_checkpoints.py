"""Retain checkpoints from an already-running pre-retention trainer, without restarting it."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dexgrasp.train.checkpoints import retain_live_checkpoint


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] in ('Z', 'X') else fields[19]
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--pid', type=int, required=True)
    args = parser.parse_args()
    identity = process_identity(args.pid)
    if identity is None:
        parser.error('Training process is not running')
    inode = None
    while True:
        finished = process_identity(args.pid) != identity
        if args.checkpoint.exists() and args.checkpoint.stat().st_ino != inode:
            destination, inode = retain_live_checkpoint(args.checkpoint)
            print(json.dumps({'retained': str(destination), 'time': time.time()}), flush=True)
        if finished:
            print(json.dumps({'status': 'training_process_exited', 'time': time.time()}), flush=True)
            break
        time.sleep(2)


if __name__ == '__main__':
    main()
