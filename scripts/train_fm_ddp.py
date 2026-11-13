"""torchrun entry point; isolate each GPU before importing Torch/MinkowskiEngine."""
import os
from pathlib import Path
import sys

if 'LOCAL_RANK' not in os.environ:
    raise SystemExit('Launch with torchrun --nproc_per_node=<GPU count> scripts/train_fm_ddp.py ...')
# Spawned DataLoader workers inherit the already isolated device list.
if os.environ.get('DEXGRASP_ISOLATED_GPU') != '1':
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    devices = visible.split(',') if visible else [str(i) for i in range(int(os.environ['LOCAL_WORLD_SIZE']))]
    os.environ['CUDA_VISIBLE_DEVICES'] = devices[int(os.environ['LOCAL_RANK'])]
    os.environ['DEXGRASP_ISOLATED_GPU'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dexgrasp.train.distributed import main

if __name__ == '__main__':
    main()
