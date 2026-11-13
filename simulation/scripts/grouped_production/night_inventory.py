"""CPU-only mode, inventory and rank identity contracts for night production."""
SCALES = ('005','008','011','014','017','020')
SCHEMAS = {'floating':'floating-night/v2', 'tabletop':'tabletop-night/v1'}

def scale_of(case, mode):
    if mode == 'tabletop':
        assert case['scale_code'] in SCALES
        assert isinstance(case['official_pose_index'], int) and case['official_pose_index'] >= 0
        return case['scale_code']
    return case['scene_id'].rsplit('_scale',1)[1]

def validate_inventory(cases, mode):
    assert mode in SCHEMAS
    assert len({c['scene_id'] for c in cases}) == len(cases), 'Duplicate scene ID'
    groups = {}
    for case in cases:
        key = (case['object_id'], case['official_pose_index'] if mode == 'tabletop' else None)
        groups.setdefault(key, []).append(scale_of(case, mode))
    assert groups and all(sorted(v) == list(SCALES) for v in groups.values()), 'Each selected pose/object requires exactly six scales'
    return groups

def validate_mode(config):
    schema = config['checkpoint_schema']
    if schema == 'tabletop-single-gpu/v1':
        assert config['world_size'] == 1 and not config.get('continuous'), 'Legacy bounded schema is single-GPU only'
        return 'tabletop'
    mode = config.get('mode', 'floating')
    assert mode in SCHEMAS and schema == SCHEMAS[mode], 'Mode/schema identity mismatch'
    assert 0 <= config['rank'] < config['world_size']
    uuids = config['rank_gpu_uuids']
    assert len(uuids) == len(set(uuids)) == config['world_size']
    assert config['gpu_uuid'] == uuids[config['rank']]
    if config['world_size'] > 1:
        from pathlib import Path
        root = Path(config['output_root']).resolve()
        assert root.name == f"rank-{config['rank']:03d}", 'Rank output namespace mismatch'
        assert Path(config['checkpoint_root']).resolve().parent == root, 'Checkpoint must be rank-local'
    return mode
