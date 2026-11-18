"""Reuse a completed deep audit only after rechecking every content digest."""
import json
from pathlib import Path
from .dataset import audit_dataset, sha256, contained, official_test_objects, SPLITS
from ..geometry.hand import URDFHand


def auditor_sources():
    root = Path(__file__).resolve().parents[1]
    paths = [root/'data/dataset.py', root/'data/audit_receipt.py',
             root/'geometry/hand.py', root/'geometry/direction.py']
    return {str(path.relative_to(root)): sha256(path) for path in paths}


def create_receipt(root, scale=.2, *, dataset_format='data_v1', official_test_root=None):
    identity = audit_dataset(root, scale, deep=True, dataset_format=dataset_format, official_test_root=official_test_root)
    hand = URDFHand(Path(root)/'hand/hand_spec.json')
    identity['hand_urdf_sha256'] = sha256(hand.urdf_path)
    return dict(schema='dexgrasp-deep-audit-v1', coord_scale_m=scale,
                auditor_sources=auditor_sources(), identity=identity)


def verified_identity(root, scale, receipt_path, *, dataset_format='data_v1', official_test_root=None):
    receipt = json.loads(Path(receipt_path).read_text())
    if (receipt.get('schema') != 'dexgrasp-deep-audit-v1' or receipt['coord_scale_m'] != scale
            or receipt['auditor_sources'] != auditor_sources()):
        raise ValueError('deep audit receipt schema, scale or auditor code changed')
    identity = receipt['identity']
    if identity.get('schema_version') != (2 if dataset_format == 'data_v2' else 1):
        raise ValueError('audit receipt has a different dataset format')
    if official_test_root and official_test_objects(official_test_root)[1] != identity.get('official_test_sha256'):
        raise ValueError('official test exclusion lists differ from the audited lists')
    if identity.get('deep') is not True:
        raise ValueError('a completed deep audit is required')
    root = Path(root)
    for relative, digest in identity.get('scene_object_files', {}).items():
        if sha256(contained(root, relative)) != digest:
            raise ValueError('audited scene object identity changed')
    if identity['hand_spec_sha256'] != sha256(root/'hand/hand_spec.json'):
        raise ValueError('audited hand spec changed')
    hand = URDFHand(root/'hand/hand_spec.json')
    if identity['hand_urdf_sha256'] != sha256(hand.urdf_path):
        raise ValueError('audited URDF changed')
    for split in SPLITS:
        if identity['split_sha256'][split] != sha256(root/'splits'/f'{split}.jsonl'):
            raise ValueError('audited split changed')
    for relative, saved in identity['shard_stats'].items():
        path = contained(root, relative)
        stat = path.stat()
        if (stat.st_size != saved['bytes'] or stat.st_mtime_ns != saved['mtime_ns']
                or sha256(path) != saved['sha256']):
            raise ValueError(f'audited shard changed: {relative}')
    return identity
