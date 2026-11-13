"""Local, foreground, guarded generation with explicit same-source resume."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def validate(config):
    root = Path(__file__).resolve().parents[1]
    if Path(config['source']).resolve() != root:
        raise ValueError('source must point to this simulation checkout')
    if config.get('world_size') != 1 or config.get('rank') != 0:
        raise ValueError('this entry supports one local GPU')
    if config.get('execution_mode') != 'sequential' or config.get('worker') != 'night_worker.py':
        raise ValueError('requires sequential night_worker.py')
    if config.get('deployment_template_only'):
        raise ValueError('provide an executable config')
    validate_execution_window(config)
    locks = [Path(p).resolve() for p in config['locks']]
    if len(set(locks)) < 3 or not all(p.is_file() for p in locks):
        raise ValueError('three distinct existing compatibility locks required')
    if config.get('eval_noslip_iterations') != 0:
        raise ValueError('production protocol uses explicit noslip0')
    scripts = root/'scripts/grouped_production'
    sys.path.insert(0, str(scripts))
    from night_config import validate_worker_mode
    from night_inventory import validate_inventory
    from night_schedule import validate_schedule
    from .preflight import inspect_runtime
    inspect_runtime()
    mode = validate_worker_mode(config)
    validate_schedule(config)
    cases = []
    for group in config['groups']:
        p = Path(group['index'])
        if hashlib.sha256(p.read_bytes()).hexdigest() != config['group_index_sha256'][str(p)]:
            raise ValueError('group index hash changed')
        cases.extend(json.loads(p.read_text())['cases'])
    validate_inventory(cases, mode)
    return root



def validate_execution_window(config, now=None):
    """Product scheduling policy, separate from any task's experimental budget."""
    now = time.time() if now is None else now
    if type(config.get('continuous', False)) is not bool:
        raise ValueError('continuous must be an explicit boolean')
    if config.get('continuous', False):
        if any(config.get(k) is not None for k in ('case_limit_s','hard_deadline_epoch','stop_new_init_epoch','tours','max_group_visits')):
            raise ValueError('continuous mode must not inherit bounded deadlines/tours')
    else:
        deadline = config.get('hard_deadline_epoch')
        if not isinstance(deadline, (int,float)) or not math.isfinite(deadline) or deadline <= now:
            raise ValueError('bounded mode requires an explicit future deadline')
        limit = config.get('case_limit_s')
        if limit is not None and (not isinstance(limit, (int,float)) or not math.isfinite(limit) or limit <= 0):
            raise ValueError('case_limit_s must be positive when supplied')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('check','run','resume','read'))
    p.add_argument('config'); p.add_argument('--name')
    a = p.parse_args(); config = json.loads(Path(a.config).read_text())
    if a.action == 'read':
        root = Path(config['checkpoint_root'])
        print((root/'summary.json').read_text()); return
    root = validate(config)
    if a.action == 'check':
        print(json.dumps(dict(valid=True, resource_admission='NOT_RUN', physics='NOT_RUN'))); return
    if not a.name or Path(a.name).name != a.name or a.name in ('.','..'):
        p.error('run/resume requires a simple unique --name')
    checkpoint = Path(config['checkpoint_root'])
    if a.action == 'run' and checkpoint.exists():
        raise ValueError('existing checkpoint requires explicit resume')
    if a.action == 'resume':
        import numpy as np
        from night_checkpoint_identity import token
        state = np.load(checkpoint/'state.npy', allow_pickle=True).item()
        if state['config_hash'] != token(config, root/'scripts/grouped_production/night_worker.py'):
            raise ValueError('resume source/config identity changed')
    raise SystemExit(subprocess.call([sys.executable, str(root/'scripts/grouped_production/run.py'),
                                     str(Path(a.config).resolve()), '--name', a.name]))


if __name__ == '__main__':
    main()
