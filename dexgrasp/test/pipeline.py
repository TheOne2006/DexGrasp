"""Sample checkpoints, geometrically decode, and report explicit evaluation coverage."""
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..models.config import STAGE_ORDER
from ..data.dataset import hdf5_dataset, audit_dataset, check_stage_order, sha256, text, checkpoint_identity_matches
from ..data.prepare import scene_meshes
from ..geometry.hand import URDFHand
from ..geometry.decode import decode_keypoints
from ..geometry.direction import official_wrist_definition, raw_anchor_direction_error
from ..geometry.mesh import measure_palm_gap
from ..geometry.palm_depth import PalmSurfaceDepth
from ..models.checkpoint import load_model
from ..models.registry import model_branch, sample_model
from ..train.runner import move_batch
from .inference import world_keypoints


@contextmanager
def new_output(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    partial.touch(exist_ok=False)
    try:
        if path.exists():
            raise FileExistsError(path)
        yield partial
        partial.rename(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def generate(checkpoint, dataset_root, output, *, split='test', steps=None, candidates=8,
             batch_size=8, device='cpu', seed=0, palm_gap_m=None, direction=None,
             encoder_factory=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    device = torch.device(device)
    model, state = load_model(checkpoint, device, encoder_factory=encoder_factory)
    data_version = state['data_identity'].get('schema_version', 1)
    dataset_format = 'data_v2' if data_version == 2 else 'data_v1'
    three_stage = dataset_format == 'data_v2'
    variant = getattr(model.cfg, 'variant', model.cfg.family)
    identity = audit_dataset(dataset_root, model.cfg.coord_scale_m, deep=True, dataset_format=dataset_format,
                             official_test_root=state.get('config', {}).get('official_test_root'))
    hand = URDFHand(Path(dataset_root)/'hand/hand_spec.json')
    identity['hand_urdf_sha256'] = sha256(hand.urdf_path)
    if not checkpoint_identity_matches(state['data_identity'], identity, dataset_root):
        raise ValueError('checkpoint and dataset identity differ')
    steps = model_branch(model.cfg.family).default_steps[model.method] if steps is None else steps
    if batch_size < 1 or candidates < 1 or steps < 1:
        raise ValueError('positive batch size, candidates and steps required')
    if palm_gap_m is not None and (not np.isfinite(palm_gap_m) or palm_gap_m < 0):
        raise ValueError('palm_gap_m must be finite and nonnegative')
    if direction is not None:
        direction = np.asarray(direction, dtype=np.float32)
        if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-8:
            raise ValueError('direction override must be a finite nonzero 3-vector')
        direction = direction/np.linalg.norm(direction)
    dataset = hdf5_dataset(dataset_format)(dataset_root, split, model.cfg.coord_scale_m)
    depth_key = dataset.depth_key
    use_depth = depth_key in model.cfg.condition_keys
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        pin_memory=device.type == 'cuda', generator=torch.Generator().manual_seed(seed))
    torch.manual_seed(seed)
    total_seconds = 0.
    try:
        with new_output(output) as temporary, h5py.File(temporary, 'w') as f:
            count = len(dataset)
            f.attrs.update(schema='dexgrasp-samples-v2' if three_stage else 'dexgrasp-samples-v1', method=model.method, split=split,
                           model_family=model.cfg.family,
                           model_variant=variant, palm_gap_conditioned=use_depth,
                           direction_conditioned='grasp_direction_cmd' in model.cfg.condition_keys,
                           checkpoint_sha256=sha256(checkpoint), dataset_root=str(Path(dataset_root).resolve()),
                           hand_spec_sha256=state['hand_spec_sha256'], hand_urdf_sha256=identity['hand_urdf_sha256'],
                           steps=steps, candidates=candidates, seed=seed,
                           physical_success_evaluated=False)
            f.attrs['condition_override'] = json.dumps(dict(palm_gap_m=palm_gap_m,
                                        direction=None if direction is None else direction.tolist()))
            depth_field = 'h32_cmd' if three_stage else 'h'
            if three_stage:
                f.attrs['stage_order'] = json.dumps(STAGE_ORDER)
                if hasattr(model.cfg, 'block_conditions'):
                    f.attrs['conditioning'] = json.dumps([asdict(block) for block in model.cfg.block_conditions])
                f.create_dataset('h32_gt', shape=(count, 1), dtype=np.float32)
            if use_depth or not three_stage:
                f.create_dataset(depth_field, shape=(count, 1), dtype=np.float32, chunks=True)
            pose_shape = (len(STAGE_ORDER), 21, 3) if three_stage else (21, 3)
            for key, shape in [('K', (count, candidates, *pose_shape)), ('d', (count, 3))]:
                f.create_dataset(key, shape=shape, dtype=np.float32, chunks=True)
            for key in ('scene_id', 'view_id', 'object_id', 'grasp_id'):
                f.create_dataset(key, shape=(count,), dtype=h5py.string_dtype('utf-8'))
            cursor = 0
            for raw in loader:
                if palm_gap_m is not None and (use_depth or not three_stage):
                    raw['condition'][depth_key] = torch.full_like(raw['condition'][depth_key], palm_gap_m)
                if direction is not None:
                    raw['condition']['grasp_direction_cmd'] = torch.from_numpy(direction).expand(len(raw['keypoints_gt']), -1)
                model_batch = {**raw, 'condition': {k: raw['condition'][k] for k in model.cfg.condition_keys}}
                batch = move_batch(model_batch, device)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                generated = sample_model(model, batch['condition'], steps=steps, candidates=candidates)
                n = len(batch['keypoints_gt'])
                if generated.shape != (n * candidates, *pose_shape):
                    raise ValueError('sampler returned incompatible candidate count or keypoint dimensions')
                world = world_keypoints(generated, batch['metadata']['center_m'], model.cfg.coord_scale_m, candidates)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                total_seconds += time.perf_counter()-started
                f['K'][cursor:cursor+n] = world.cpu().numpy()
                f['d'][cursor:cursor+n] = raw['condition']['grasp_direction_cmd'].numpy()
                if depth_field in f:
                    f[depth_field][cursor:cursor+n] = raw['condition'][depth_key].numpy()
                if three_stage:
                    f['h32_gt'][cursor:cursor+n] = raw['h32'].numpy()
                for key in ('scene_id', 'view_id', 'object_id', 'grasp_id'):
                    f[key][cursor:cursor+n] = raw['metadata'][key]
                cursor += n
            f.attrs['encoding_and_generation_seconds'] = total_seconds
    finally:
        dataset.close()
    return dict(output=str(output), method=model.method, model_variant=variant,
                samples=len(dataset), candidates=candidates,
                nfe_per_candidate=steps, encoding_and_generation_seconds=total_seconds,
                note='Timing excludes SVD/IK, collision checks and physics; physical success is unmeasured.')


def decode_stages(keypoints, hand, direction, depth_mesh, requested_depth, *, tolerance_m, max_nfev):
    stages = {}
    for name, points in zip(STAGE_ORDER, keypoints, strict=True):
        try:
            decoded = decode_keypoints(points, hand, direction, tolerance_m=tolerance_m, max_nfev=max_nfev)
            stages[name] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in decoded.items()}
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            stages[name] = dict(ik_success=False, joint_limits_valid=False, error=str(exc))
    grasp = stages['grasp']
    raw_angle = raw_anchor_direction_error(keypoints[STAGE_ORDER.index('grasp')], hand, direction)
    result = dict(stages=stages, ik_success=all(s['ik_success'] for s in stages.values()),
                  joint_limits_valid=all(s['joint_limits_valid'] for s in stages.values()),
                  direction_error_deg=grasp.get('direction_error_deg'), raw_direction_error_deg=raw_angle,
                  palm_gap_valid=False)
    if 'R' in grasp:
        official = np.asarray(grasp['R']) @ np.asarray(hand.spec['official_from_grasp_rotation']).T
        depth = float(depth_mesh.measure(official, grasp['t']))
        if np.isfinite(depth):
            result.update(palm_gap_valid=True, measured_h32_m=depth, palm_gap_error_m=abs(depth-requested_depth))
    return result


def decode_file(samples, output, *, tolerance_m=0.003, max_nfev=100):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    with h5py.File(samples, 'r') as f:
        three_stage = f.attrs.get('schema') == 'dexgrasp-samples-v2'
        if three_stage:
            check_stage_order(f.attrs.get('stage_order'))
            conditioning = json.loads(f.attrs['conditioning']) if 'conditioning' in f.attrs else None
            if f['K'].ndim != 5 or f['K'].shape[2:] != (3, 21, 3):
                raise ValueError('invalid three-stage sample dimensions')
        elif f.attrs.get('schema') != 'dexgrasp-samples-v1':
            raise ValueError('invalid sample file')
        root = Path(f.attrs['dataset_root'])
        hand = URDFHand(root/'hand/hand_spec.json')
        if three_stage:
            hand.spec['direction_condition'] = official_wrist_definition(hand.spec)
        if sha256(hand.spec_path) != f.attrs['hand_spec_sha256'] or sha256(hand.urdf_path) != f.attrs['hand_urdf_sha256']:
            raise ValueError('hand calibration changed since sampling')
        scenes = {}
        for path in (root/'scenes').glob('*.json'):
            scene = json.loads(path.read_text())
            scenes[(scene['scene_id'], scene.get('view_id'))] = scene
        # A failed decode never publishes a truncated candidate set as a completed run.
        with new_output(output) as temporary, temporary.open('w') as stream:
            cached_key, target = None, None
            for i in range(len(f['K'])):
                key = (text(f['scene_id'][i]), text(f['view_id'][i]))
                if key != cached_key:
                    scene = scenes.get(key) or scenes[(key[0], None)]
                    if three_stage:
                        target = PalmSurfaceDepth(*scene_meshes(root, scene, target_only=True), scale=1.)
                    else:
                        target, _ = scene_meshes(root, scene)
                    cached_key = key
                for j, keypoints in enumerate(f['K'][i]):
                    started = time.perf_counter()
                    row = dict(sample_index=i, candidate_index=j, scene_id=key[0], view_id=key[1],
                               model_variant=text(f.attrs.get('model_variant', 'full')),
                               palm_gap_conditioned=bool(f.attrs.get('palm_gap_conditioned', True)),
                               direction_conditioned=bool(f.attrs.get('direction_conditioned', True)),
                               grasp_id=text(f['grasp_id'][i]),
                               physical_success=None, collision_free=None)
                    if three_stage:
                        label = float(f['h32_gt'][i, 0])
                        command = float(f['h32_cmd'][i, 0]) if row['palm_gap_conditioned'] else None
                        row.update(h32_gt_m=label, requested_h32_m=command, direction_projected=False)
                        if conditioning is not None:
                            row['conditioning'] = conditioning
                        row.update(decode_stages(keypoints, hand, f['d'][i], target, label if command is None else command,
                                                 tolerance_m=tolerance_m, max_nfev=max_nfev))
                    else:
                        row['requested_palm_gap_m'] = float(f['h'][i, 0])
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
    stage_metrics = {}
    if 'stages' in rows[0]:
        for metric, key in [('raw_direction_pass_rate', 'raw_direction_error_deg'),
                            ('decoded_direction_pass_rate', 'direction_error_deg')]:
            stage_metrics[metric] = sum(r.get(key) is not None and r[key] <= direction_tolerance_deg for r in rows) / n
        stage_metrics['stage_ik_success_rate'] = {
            stage: sum(r['stages'][stage]['ik_success'] for r in rows) / n for stage in STAGE_ORDER
        }
    return dict(candidates=n, **stage_metrics, gap_tolerance_m=gap_tolerance_m, direction_tolerance_deg=direction_tolerance_deg,
                gap_measurement_coverage=len(gaps)/n, gap_error_mm_quantiles=
                dict(zip(('p50', 'p90', 'p95'), np.quantile(gaps, [.5, .9, .95]).tolist())) if gaps else None,
                direction_error_deg_mean=float(np.mean(angles)) if angles else None,
                geometric_condition_pass_rate=geometric/n, physics_evaluation_coverage=measured/n,
                physical_success_rate=physical/n if measured == n else None,
                joint_success_rate=verified/n if measured == n else None,
                verified_joint_success_count=verified,
                note='Invalid gap/IK candidates stay in the denominator. Unmeasured physics is not success.')
