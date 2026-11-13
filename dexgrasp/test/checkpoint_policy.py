"""Training-run-scoped exclusions, independent of test performance."""
import json
from pathlib import Path
import re


def exclusion(checkpoint, step=None):
    path = Path(checkpoint).resolve()
    manifest = path.parent.parent/'evaluation_policy.json'
    if not manifest.is_file():
        return None
    policy = json.loads(manifest.read_text())
    if policy['schema'] != 'dexgrasp-evaluation-eligibility-v1':
        raise ValueError('Unknown checkpoint eligibility policy')
    rule = policy['runs'].get(path.parent.name)
    if rule is None:
        return None
    match = re.fullmatch(r'step_(\d+)\.pt', path.name)
    recorded_step = int(match[1]) if match else step
    if path.name in rule.get('excluded_aliases', []) or (recorded_step is not None and recorded_step > rule['max_step']):
        return dict(reason='post_training_instability', max_step=rule['max_step'], policy=str(manifest))
    if recorded_step is None:
        raise ValueError('Cannot establish checkpoint step under evaluation policy: '+str(path))
    return None


def require_eligible(checkpoint, step=None):
    reason = exclusion(checkpoint, step)
    if reason:
        raise ValueError(f"Checkpoint excluded from evaluation after training instability: {checkpoint} (last eligible step {reason['max_step']})")
