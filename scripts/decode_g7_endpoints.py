"""Decode two generated endpoints and retain the frozen middle pose and draw ids."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from evaluate_g7_anchors import BASE, FROZEN, source
from dexgrasp.test.formal import official_hand
from dexgrasp.test.frozen import digest, load_decoded_shard, save
from dexgrasp.test.metrics import decode_metrics


def pack(rotation, translation, joints):
    quat = Rotation.from_matrix(rotation @ frame.T).as_quat(scalar_first=True)
    return np.r_[translation, quat, joints]


def decode(block):
    output = np.full((len(block), 3, 29), np.nan, dtype='f4')
    for j, i in enumerate(block):
        mid = anchors[index['index'][i] // 2]
        if mid['decoded']:
            output[j, 1] = pack(mid['R'], mid['t'] + centers[i], mid['q'])
        for stage, k in [(0, 0), (2, 1)]:
            p = decode_metrics(keys[i, k], hand, index['directions'][i])
            if p['decoded'] and p['joint_limits_valid']:
                output[j, stage] = pack(p['R'], p['t'] + centers[i], p['q'])
    return block, output


def main(args):
    global hand, frame, keys, index, anchors, centers
    parent = json.loads((FROZEN / 'plan.json').read_text())
    data = Path(parent['root'])
    hand = official_hand(data / 'hand/hand_spec.json')
    frame = np.array(hand.spec['official_from_grasp_rotation'])
    observations = np.load(data / 'observation_index.npy', mmap_mode='r')
    origins = np.load(FROZEN / 'quality/centers.npy')
    for suite in args.suites:
        root, mode = source(suite)
        for g in args.groups:
            for rank in (0, 1):
                folder = args.output / suite / g / f'rank{rank}'
                receipt = folder / 'decode.json'
                protocol = dict(generation=digest(folder / 'generation.json'), code=digest(__file__))
                if receipt.exists():
                    if json.loads(receipt.read_text())['protocol'] != protocol:
                        raise RuntimeError(f'Decode inputs changed in {folder}')
                    continue
                keys = np.load(folder / 'endpoints.npy', mmap_mode='r')
                with np.load(folder / 'index.npz') as z:
                    index = dict(z)
                anchors = load_decoded_shard(root / g / f'rank{rank}' / mode)
                centers = (np.zeros((len(keys), 3)) if suite == 'official'
                           else origins[observations[index['packed']]])
                path = folder / 'qpos.npy'
                result = np.lib.format.open_memmap(path, mode='w+', dtype='f4', shape=(len(keys), 3, 29))
                tick = time.perf_counter()
                with ProcessPoolExecutor(args.workers) as pool:
                    blocks = (np.arange(i,min(i+64,len(keys))) for i in range(0,len(keys),64))
                    for block, values in pool.map(decode, blocks):
                        result[block] = values
                result.flush()
                save(receipt, dict(protocol=protocol, count=len(keys), sha256=digest(path),
                                   decoded=int(np.isfinite(result).all(axis=(1,2)).sum()),
                                   seconds=time.perf_counter()-tick))
                print(suite, g, rank, len(keys), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--suites', nargs='+', required=True)
    p.add_argument('--groups', default='ABCD')
    p.add_argument('--workers', type=int, default=24)
    main(p.parse_args())
