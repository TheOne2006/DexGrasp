"""Copy a dataset, relabel only d to official +Zh, and audit every sample."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dexgrasp.data.dataset import audit_dataset, sha256
from dexgrasp.geometry.direction import WRIST_Z, hand_direction
from dexgrasp.train.runner import source_state


def migrate(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    before = audit_dataset(source)
    # Full copies: future writes cannot modify a historical dataset via hardlinks.
    shutil.copytree(source, destination)
    (destination/'preparation.json').unlink()  # Unpublished until the audit passes.
    spec_path = destination/'hand/hand_spec.json'
    spec = json.loads(spec_path.read_text())
    spec['label_version'] = 'kinematic-wrist-z-v2'
    spec['direction_condition'] = dict(kind=WRIST_Z, local_axis=[0, 1, 0],
                                      official_axis='+Zh', sign='wrist toward fingers')
    spec_path.write_text(json.dumps(spec, indent=2)+'\n')
    count = 0
    for relative in before['shard_stats']:
        with h5py.File(destination/relative, 'r+') as new, h5py.File(source/relative, 'r') as old:
            rotation = new['grasps/R'][:]
            direction = hand_direction(rotation, spec).astype(np.float32)
            new['grasps/d'][:] = direction
            new.attrs['hand_spec_sha256'] = sha256(spec_path)
            new.attrs['label_version'] = spec['label_version']
            new.attrs['direction_condition'] = WRIST_Z
            for group in ('observations', 'grasps'):
                for key in old[group]:
                    if (group, key) != ('grasps', 'd') and not np.array_equal(old[group][key][:], new[group][key][:]):
                        raise ValueError(f'Unexpected change: {group}/{key}')
            if not np.allclose(np.sum(old['grasps/d'][:]*direction, axis=-1), 0., atol=2e-6):
                raise ValueError('Expected old and new directions to be perpendicular')
            count += len(direction)
    after = audit_dataset(destination)
    if audit_dataset(source) != before:
        raise ValueError('Historical source dataset changed')
    record = dict(created_at=datetime.now(timezone.utc).isoformat(), source_dataset=str(source),
                  dataset=str(destination), direction_definition=spec['direction_condition'],
                  changed='only hand metadata and grasp direction labels; no pose retargeting or point resampling',
                  preserved='all observations, R, t, q, K, h, hit_ratio, split rows, meshes and raw grasps',
                  samples_relabelled=count, old_new_angle_deg=90., before=before, after=after,
                  source=source_state(), script_sha256=sha256(__file__))
    (destination/'direction_migration.json').write_text(json.dumps(record, indent=2)+'\n')
    summary=json.loads((source/'preparation.json').read_text())
    summary.update(dataset=destination.name, audit=after, direction_definition=spec['direction_condition'],
                   direction_migration='direction_migration.json')
    (destination/'preparation.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(dict(dataset=str(destination), samples=count, audit=after), indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args=parser.parse_args()
    migrate(args.source, args.destination)
