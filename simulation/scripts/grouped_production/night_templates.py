"""Install a frozen, validated cumulative template snapshot into a fresh checkpoint.

Snapshot layout and exact-content identity follow eight_gpu_benchmark/templates.py.
Only trusted project NPY bundles are accepted. This performs no physics validation.
"""
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install_snapshot(config, state, root):
    snapshot = config.get('template_snapshot')
    if not snapshot:
        return
    root = Path(root)
    source = Path(snapshot)
    manifest = source / 'manifest.json'
    expected = config['template_manifest_sha256']
    assert sha(manifest) == expected, 'Template snapshot identity changed'
    m = json.loads(manifest.read_text())
    assert m['status'] == 'PASS' and m['seed_count'] == len(m['seed_hashes']) == 35
    for family, expected_seed in m['seed_hashes'].items():
        assert sha(Path(config['asset_root'])/'hand/sharpa_v2/init_tmpl'/f'{family}.npy') == expected_seed
    if 'initial_template_snapshot' in state:
        assert state['initial_template_snapshot']['manifest_sha256'] == expected
        return
    assert not state['completed'] and not state['family_archive'], 'Import only into fresh state'
    destination = root/'seed-history'
    destination.mkdir(exist_ok=True)
    for rel, expected_bundle in m['bundle_hashes'].items():
        assert Path(rel).name == rel and rel.endswith('.npy')
        assert sha(source/rel) == expected_bundle
        target = destination/rel
        if target.exists():
            assert sha(target) == expected_bundle
        else:
            temp = target.with_suffix('.partial')
            shutil.copyfile(source/rel, temp)
            assert sha(temp) == expected_bundle
            os.replace(temp, target)
        state.setdefault('artifacts', {})['seed-history/'+rel] = expected_bundle
    ids = []
    for family, refs in m['references'].items():
        assert family in m['seed_hashes']
        for ref in refs:
            assert ref['path'] == 'seed-history/'+family+'.npy'
            assert family+'.npy' in m['bundle_hashes']
            ids.append(ref['id'])
    assert len(ids) == len(set(ids)) == m['content_unique']
    state['family_archive'] = m['references']
    state['historical_template_ids'] = ids
    state['feedback_seen_ids'] = ids
    state['initial_template_snapshot'] = dict(path=str(source), manifest_sha256=expected,
                                              history=len(ids), seeds=35)
    shutil.copyfile(manifest, destination/'manifest.json')
    state['artifacts']['seed-history/manifest.json'] = expected
