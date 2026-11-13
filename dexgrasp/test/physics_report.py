"""Read physical reports with candidate identity and unknowns preserved."""
import hashlib
import json
from pathlib import Path


def summarize(decoded, report):
    raw = Path(decoded).read_bytes()
    candidates = [json.loads(x) for x in raw.splitlines() if x.strip()]
    ids = {(r['sample_index'], r['candidate_index']) for r in candidates}
    if not candidates or len(ids) != len(candidates):
        raise ValueError('empty/duplicate decoded identities')
    result = json.loads(Path(report).read_text())
    if result.get('schema') != 'dexgrasp-physics-result/v1' or result.get('decoded_sha256') != hashlib.sha256(raw).hexdigest():
        raise ValueError('physics report does not match decoded input')
    if not result.get('protocol') or result.get('backend') not in ('cpu', 'warp'):
        raise ValueError('missing physics protocol/backend')
    seen, counts = set(), dict(success=0, failure=0, invalid_input=0, simulation_error=0)
    for row in result['results']:
        key = row['sample_index'], row['candidate_index']
        if key not in ids or key in seen:
            raise ValueError('unknown/duplicate physical candidate')
        seen.add(key)
        status = row['status']
        expected = {'success': True, 'failure': False, 'invalid_input': None, 'simulation_error': None}
        if status not in expected or row['physical_success'] is not expected[status]:
            raise ValueError('inconsistent status/physics result')
        counts[status] += 1
    measured = counts['success'] + counts['failure']
    return dict(candidates=len(ids), protocol=result['protocol'], backend=result['backend'], counts=counts,
                missing=len(ids-seen), physics_coverage=measured/len(ids),
                verified_physical_success_count=counts['success'],
                physical_success_rate=counts['success']/len(ids) if measured == len(ids) else None,
                joint_success_rate=None,
                note='Physics-only report. No collision-free or legacy ray-gap joint acceptance is inferred.')
