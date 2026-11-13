"""Sample checkpoints, geometrically decode, and report explicit evaluation coverage."""
import json
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..data.dataset import GraspDataset, audit_dataset, sha256, text, checkpoint_identity_matches
from ..data.prepare import scene_meshes
from ..geometry.hand import URDFHand
from ..geometry.decode import decode_keypoints
from ..geometry.mesh import measure_palm_gap
from ..models.encoders import GeometryEncoder
from ..train.objectives import sample_keypoints
from ..train.runner import load_model, move_batch


def generate(checkpoint, dataset_root, output, *, split='test', steps=None, candidates=8,
             batch_size=8, device='cpu', seed=0, palm_gap_m=None, direction=None,
             encoder_factory=GeometryEncoder):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    device = torch.device(device)
    model, state = load_model(checkpoint, device, encoder_factory=encoder_factory)
    identity = audit_dataset(dataset_root, model.cfg.coord_scale_m, deep=True)
    hand = URDFHand(Path(dataset_root)/'hand/hand_spec.json')
    identity['hand_urdf_sha256'] = sha256(hand.urdf_path)
    if not checkpoint_identity_matches(state['data_identity'], identity, dataset_root):
        raise ValueError('checkpoint and dataset identity differ')
    steps = (10 if model.method == 'fm' else 1) if steps is None else steps
    if batch_size < 1 or candidates < 1 or steps < 1:
        raise ValueError('positive batch size, candidates and steps required')
    if palm_gap_m is not None and (not np.isfinite(palm_gap_m) or palm_gap_m < 0):
        raise ValueError('palm_gap_m must be finite and nonnegative')
    if direction is not None:
        direction = np.asarray(direction, dtype=np.float32)
        if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-8:
            raise ValueError('direction override must be a finite nonzero 3-vector')
        direction = direction/np.linalg.norm(direction)
    dataset = GraspDataset(dataset_root, split, model.cfg.coord_scale_m)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        pin_memory=device.type == 'cuda', generator=torch.Generator().manual_seed(seed))
    torch.manual_seed(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name+'.partial')
    if temporary.exists():
        raise FileExistsError(temporary)
    total_seconds = 0.
    try:
        with h5py.File(temporary, 'x') as f:
            count = len(dataset)
            f.attrs.update(schema='dexgrasp-samples-v1', method=model.method, split=split,
                           model_variant=model.cfg.variant, palm_gap_conditioned=model.cfg.use_gap,
                           checkpoint_sha256=sha256(checkpoint), dataset_root=str(Path(dataset_root).resolve()),
                           hand_spec_sha256=state['hand_spec_sha256'], hand_urdf_sha256=identity['hand_urdf_sha256'],
                           steps=steps, candidates=candidates, seed=seed,
                           physical_success_evaluated=False)
            f.attrs['condition_override'] = json.dumps(dict(palm_gap_m=palm_gap_m,
                                        direction=None if direction is None else direction.tolist()))
            for key, shape in [('K', (count, candidates, 21, 3)), ('d', (count, 3)), ('h', (count, 1))]:
                f.create_dataset(key, shape=shape, dtype=np.float32, chunks=True)
            for key in ('scene_id', 'view_id', 'object_id', 'grasp_id'):
                f.create_dataset(key, shape=(count,), dtype=h5py.string_dtype('utf-8'))
            cursor = 0
            for raw in loader:
                if palm_gap_m is not None:
                    raw['condition']['palm_gap'] = torch.full_like(raw['condition']['palm_gap'], palm_gap_m)
                if direction is not None:
                    raw['condition']['grasp_direction_cmd'] = torch.from_numpy(direction).expand(len(raw['keypoints_gt']), -1)
                model_batch = {**raw, 'condition': {k: raw['condition'][k] for k in model.cfg.condition_keys}}
                batch = move_batch(model_batch, device)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                generated = sample_keypoints(model, batch['condition'], steps=steps, candidates=candidates)
                n = len(batch['keypoints_gt'])
                generated = generated.reshape(n, candidates, 21, 3)
                world = generated*model.cfg.coord_scale_m + batch['metadata']['center_m'][:, None, None]
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                total_seconds += time.perf_counter()-started
                f['K'][cursor:cursor+n] = world.cpu().numpy()
                f['d'][cursor:cursor+n] = raw['condition']['grasp_direction_cmd'].numpy()
                f['h'][cursor:cursor+n] = raw['condition']['palm_gap'].numpy()
                for key in ('scene_id', 'view_id', 'object_id', 'grasp_id'):
                    f[key][cursor:cursor+n] = raw['metadata'][key]
                cursor += n
            f.attrs['encoding_and_generation_seconds'] = total_seconds
        temporary.rename(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        dataset.close()
    return dict(output=str(output), method=model.method, model_variant=model.cfg.variant,
                samples=len(dataset), candidates=candidates,
                nfe_per_candidate=steps, encoding_and_generation_seconds=total_seconds,
                note='Timing excludes SVD/IK, collision checks and physics; physical success is unmeasured.')


def decode_file(samples, output, *, tolerance_m=0.003, max_nfev=100):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(output.name+'.partial')
    if temporary.exists():
        raise FileExistsError(temporary)
    with h5py.File(samples, 'r') as f:
        if f.attrs.get('schema') != 'dexgrasp-samples-v1':
            raise ValueError('invalid sample file')
        root = Path(f.attrs['dataset_root'])
        hand = URDFHand(root/'hand/hand_spec.json')
        if sha256(hand.spec_path) != f.attrs['hand_spec_sha256'] or sha256(hand.urdf_path) != f.attrs['hand_urdf_sha256']:
            raise ValueError('hand calibration changed since sampling')
        scenes = {}
        for path in (root/'scenes').glob('*.json'):
            scene = json.loads(path.read_text())
            scenes[(scene['scene_id'], scene['view_id'])] = scene
        output.parent.mkdir(parents=True, exist_ok=True)
        # A failed decode never publishes a truncated candidate set as a completed run.
        with temporary.open('x') as stream:
            cached_key, target = None, None
            for i in range(len(f['K'])):
                key = (text(f['scene_id'][i]), text(f['view_id'][i]))
                if key != cached_key:
                    target, _ = scene_meshes(root, scenes[key])
                    cached_key = key
                for j, keypoints in enumerate(f['K'][i]):
                    started = time.perf_counter()
                    row = dict(sample_index=i, candidate_index=j, scene_id=key[0], view_id=key[1],
                               model_variant=text(f.attrs.get('model_variant', 'full')),
                               palm_gap_conditioned=bool(f.attrs.get('palm_gap_conditioned', True)),
                               grasp_id=text(f['grasp_id'][i]), requested_palm_gap_m=float(f['h'][i, 0]),
                               physical_success=None, collision_free=None)
                    try:
                        decoded = decode_keypoints(keypoints, hand, f['d'][i],
                                                   tolerance_m=tolerance_m, max_nfev=max_nfev)
                        gap = measure_palm_gap(*target, decoded['R'], decoded['t'], hand.spec['palm'])
                        row.update({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in decoded.items()})
                        row.update(palm_gap_valid=gap['valid'], measured_palm_gap_m=gap['h'],
                                   palm_gap_error_m=abs(gap['h']-row['requested_palm_gap_m']) if gap['valid'] else None,
                                   palm_hit_ratio=gap['hit_ratio'], palm_gap_invalid_reason=gap['reason'])
                    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                        row.update(ik_success=False, joint_limits_valid=False, palm_gap_valid=False, error=str(exc))
                    row['decode_seconds'] = time.perf_counter()-started
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
        temporary.rename(output)
    return dict(output=str(output), physical_success_evaluated=False)


def evaluate(decoded, *, gap_tolerance_m=0.005, direction_tolerance_deg=10., verdicts=None):
    if gap_tolerance_m <= 0 or not 0 < direction_tolerance_deg <= 180:
        raise ValueError('invalid evaluation tolerances')
    rows = [json.loads(line) for line in Path(decoded).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('empty decoded candidate file')
    identities = {(r['sample_index'], r['candidate_index']) for r in rows}
    if len(identities) != len(rows):
        raise ValueError('duplicate decoded candidate identity')
    by_id = {}
    if verdicts:
        decoded_hash = sha256(decoded)
        for line in Path(verdicts).read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            key = item['sample_index'], item['candidate_index']
            if key in by_id or key not in identities:
                raise ValueError('duplicate or unknown evaluator candidate')
            if not item.get('protocol_version') or not item.get('evaluator'):
                raise ValueError('external verdicts need evaluator and protocol_version')
            if item.get('decoded_sha256') != decoded_hash:
                raise ValueError('external verdicts belong to a different decoded candidate file')
            if type(item.get('collision_free')) is not bool or type(item.get('physical_success')) is not bool:
                raise ValueError('external collision/physics results must be booleans')
            by_id[key] = item
    gaps, angles, geometric, verified, physical, measured = [], [], 0, 0, 0, 0
    for row in rows:
        gap = row.get('palm_gap_error_m')
        angle = row.get('direction_error_deg')
        if row.get('palm_gap_valid') and gap is not None and np.isfinite(gap):
            gaps.append(float(gap)*1000)
        if angle is not None and np.isfinite(angle):
            angles.append(float(angle))
        ok = (row.get('ik_success') is True and row.get('joint_limits_valid') is True
              and row.get('palm_gap_valid') is True and gap is not None and gap <= gap_tolerance_m
              and angle is not None and angle <= direction_tolerance_deg)
        geometric += int(ok)
        verdict = by_id.get((row['sample_index'], row['candidate_index']))
        if verdict:
            measured += 1
            physical += int(verdict['physical_success'])
            verified += int(ok and verdict['collision_free'] and verdict['physical_success'])
    n = len(rows)
    return dict(candidates=n, gap_tolerance_m=gap_tolerance_m, direction_tolerance_deg=direction_tolerance_deg,
                gap_measurement_coverage=len(gaps)/n, gap_error_mm_quantiles=
                dict(zip(('p50', 'p90', 'p95'), np.quantile(gaps, [.5, .9, .95]).tolist())) if gaps else None,
                direction_error_deg_mean=float(np.mean(angles)) if angles else None,
                geometric_condition_pass_rate=geometric/n, physics_evaluation_coverage=measured/n,
                physical_success_rate=physical/n if measured == n else None,
                joint_success_rate=verified/n if measured == n else None,
                verified_joint_success_count=verified,
                note='Invalid gap/IK candidates stay in the denominator. Unmeasured physics is not success.')
