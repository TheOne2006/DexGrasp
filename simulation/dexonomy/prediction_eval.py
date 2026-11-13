"""Explicit prediction -> physical evaluation boundary; imports stay lightweight."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time


def load_request(path):
    import numpy as np
    path = Path(path).resolve()
    request = json.loads(path.read_text())
    if (request['schema'] != 'dexgrasp-physics/v1' or request['frame'] != 'world'
            or request['units'] != {'length': 'm', 'angle': 'rad'}
            or request['protocol'] not in ('floating', 'tabletop')
            or request.get('hand_pose_frame') != 'official_hand_root'):
        raise ValueError('unsupported schema/frame/units/protocol')
    if not request.get('control_source'):
        raise ValueError('control construction provenance required')
    names = request['joint_names']
    if len(names) != 22 or len(set(names)) != 22 or not all(isinstance(n, str) and n for n in names):
        raise ValueError('22 unique named Sharpa joints required')
    digest = request['decoded_sha256']
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('decoded_sha256 must be a SHA256')
    request['hand_xml'] = str((path.parent / request['hand_xml']).resolve())
    seen = set()
    if not request['candidates']:
        raise ValueError('empty candidate set')
    for row in request['candidates']:
        key = (row['sample_index'], row['candidate_index'])
        if any(type(k) is not int or k < 0 for k in key) or key in seen:
            raise ValueError('invalid or duplicate candidate identity')
        seen.add(key)
        row['scene'] = str((path.parent / row['scene']).resolve())
        if row.get('invalid_reason'):
            continue
        for stage in ('pregrasp', 'grasp', 'squeeze'):
            poses = row[stage] if isinstance(row[stage], list) else [row[stage]]
            if not poses:
                raise ValueError('empty control trajectory')
            for pose in poses:
                R, t, q = (np.asarray(pose[k], dtype=float) for k in ('R', 't', 'q'))
                if R.shape != (3, 3) or t.shape != (3,) or q.shape != (22,):
                    raise ValueError('R/t/q shape must be 3x3/3/22')
                if not all(np.isfinite(x).all() for x in (R, t, q)):
                    raise ValueError('nonfinite pose')
                if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1, atol=1e-5):
                    raise ValueError('R must be a proper rotation')
    return request


def candidate_data(row, names, host):
    import mujoco
    import numpy as np
    # The compiler order is authoritative; object joints are excluded by qpos adr.
    model = host._model
    joints = [j for j in range(model.njnt) if 7 <= model.jnt_qposadr[j] < 29]
    prefix = host._cfg.hand_prefix
    actual = [model.joint(j).name.removeprefix(prefix) for j in joints]
    if len(joints) != 22 or set(actual) != set(names):
        raise ValueError('hand model and supplied joint names differ')
    order = [names.index(n) for n in actual]
    result = dict(scene_path=row['scene'], pipeline_stage='stability_candidate')
    for stage in ('pregrasp', 'grasp', 'squeeze'):
        poses = row[stage] if isinstance(row[stage], list) else [row[stage]]
        trajectory = []
        for p in poses:
            q = np.asarray(p['q'], float)[order]
            for j, value in zip(joints, q):
                if stage == 'grasp' and model.jnt_limited[j] and not model.jnt_range[j, 0] <= value <= model.jnt_range[j, 1]:
                    raise ValueError(f'{stage}: joint outside model limits')
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, np.asarray(p['R'], float).reshape(9))
            trajectory.append(np.concatenate([p['t'], quat, q]))
        result[stage+'_qpos'] = np.stack(trajectory)
    return result


def floating_cpu(host, scene, data):
    """Six independent resets/preambles; identical fixed holds and pose criteria."""
    import mujoco
    import numpy as np
    from .op.eval_force_closure_warp import FORCE_DIRECTIONS, _candidate_trajectory, _interpolate_segment
    from .util.np_util import np_get_delta_pose
    controls, types, target = _candidate_trajectory(host, scene, data, False)
    directions = []
    def step(n):
        for _ in range(n):
            mujoco.mj_step(host._model, host._data)
            if not np.isfinite(host._data.qpos).all() or not np.isfinite(host._data.qvel).all() or host._data.warning.number.any():
                raise RuntimeError('MuJoCo nonfinite state or warning')
    for direction in FORCE_DIRECTIONS:
        host.reset_qpos(controls[0])
        for i, (a, b) in enumerate(zip(controls[:-1], controls[1:])):
            for q in _interpolate_segment(a, b, types[i], 5):
                host.set_ctrl(q, types[i]); step(5)
        checks = []
        # Match the fixed GPU force-update cadence: one update per five steps.
        for _ in range(10):
            for _ in range(10):
                host.set_active_obj_ext_fdir(direction); step(5)
            trans, rot = np_get_delta_pose(target, host.get_active_obj_pose())
            checks.append(dict(translation_m=float(trans), rotation_deg=float(rot), passes=bool(trans < .05 and rot < 15)))
        directions.append(dict(direction=direction.tolist(), passes=all(c['passes'] for c in checks), pose_checks=checks))
    return dict(passes=all(d['passes'] for d in directions), direction_results=directions)


def evaluate(request, *, backend='cpu', batch_size=6):
    import importlib.metadata
    import numpy as np
    if backend not in ('cpu', 'warp') or batch_size < 1:
        raise ValueError('invalid backend/batch size')
    if importlib.metadata.version('mujoco') != '3.17.0':
        raise ValueError('requires mujoco==3.17.0')
    from .op.eval_force_closure_warp import _make_environment, _prepare_scene
    from .tabletop_protocol import make_host, model_identity, evaluate as tabletop_evaluate
    rows, hosts = [], {}
    started = time.perf_counter()
    for row in request['candidates']:
        result = {k: row[k] for k in ('sample_index', 'candidate_index')}
        result.update(physical_success=None, collision_free=None, status='invalid_input')
        if row.get('invalid_reason'):
            result['reason'] = row['invalid_reason']; rows.append(result); continue
        try:
            scene_path = row['scene']
            if scene_path not in hosts:
                if request['protocol'] == 'tabletop':
                    host = make_host(scene_path, request['hand_xml']); scene = host.tabletop_scene
                else:
                    scene = _prepare_scene(scene_path, .1)
                    host = _make_environment(scene, request['hand_xml'], (.6, .02), False, 0, 'native')
                    host._model.opt.tolerance = 1e-8; host._model.opt.noslip_tolerance = 1e-6
                hosts[scene_path] = (host, scene)
            host, scene = hosts[scene_path]
            data = candidate_data(row, request['joint_names'], host)
            result['model_sha256'] = model_identity(host._model)
            result['scene_sha256'] = hashlib.sha256(Path(scene_path).read_bytes()).hexdigest()
        except (ValueError, KeyError, OSError) as exc:
            result['reason'] = str(exc); rows.append(result); continue
        result['status'] = 'simulation_error'
        try:
            if request['protocol'] == 'tabletop':
                report = tabletop_evaluate([{'id': str(len(rows)), 'data': data}], [host], [0],
                                           backend=backend, batch_size=batch_size)
                metrics = dict(slide=report['slide'], direction_results=report['directions'], combined=report['combined'])
                passed = report['counts']['joint_pass'] == 1
            elif backend == 'cpu':
                metrics = floating_cpu(host, scene, data); passed = metrics['passes']
            else:
                from omegaconf import OmegaConf
                from .op.eval_force_closure_warp_mixed import evaluate_force_closure_data_warp_mixed
                cfg = OmegaConf.create(dict(hand=dict(xml_path=request['hand_xml']), obj_mass=.1,
                    physics_profile='native', warp_device='cuda:0', op=dict(miu_coef=[.6,.02],
                    hard=False, noslip_iterations=0, skip_pregrasp=False, trans_thre=.05, rot_thre=15)))
                report = evaluate_force_closure_data_warp_mixed([data], cfg, source_names=[str(len(rows))], world_batch_size=batch_size)
                metrics = report['results'][0]['metrics']; passed = report['results'][0]['success']
            result.update(status='success' if passed else 'failure', physical_success=bool(passed), metrics=metrics)
        except (ValueError, RuntimeError) as exc:
            result['reason'] = str(exc)
        rows.append(result)
    measured = sum(r['physical_success'] is not None for r in rows)
    return dict(schema='dexgrasp-physics-result/v1', decoded_sha256=request['decoded_sha256'],
                protocol=request['protocol']+'-noslip0/v1', backend=backend,
                control_source=request['control_source'], results=rows,
                counts={s: sum(r['status'] == s for r in rows) for s in ('success','failure','invalid_input','simulation_error')},
                candidates=len(rows), physics_coverage=measured/len(rows),
                physical_success_rate=sum(r['physical_success'] is True for r in rows)/len(rows) if measured == len(rows) else None,
                elapsed_s=time.perf_counter()-started,
                hand_xml_sha256=hashlib.sha256(Path(request['hand_xml']).read_bytes()).hexdigest(),
                runtime_sources={str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in Path(__file__).parent.rglob('*.py')},
                versions={n: importlib.metadata.version(n) for n in
                    (('numpy','mujoco','mujoco-warp','warp-lang') if backend == 'warp' else ('numpy','mujoco'))},
                collision_note='Collision-free was not independently measured; null is not true.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('request'); p.add_argument('--output', required=True)
    p.add_argument('--backend', choices=('cpu','warp'), default='cpu')
    p.add_argument('--batch-size', type=int, default=6)
    p.add_argument('--trusted-scenes', action='store_true', help='Authorize reading local pickle-backed Dexonomy scene files')
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    request = load_request(a.request)
    if a.check_only:
        result = dict(valid=True, candidates=len(request['candidates']), physics='NOT_RUN')
    else:
        if not a.trusted_scenes:
            p.error('existing Dexonomy scenes use pickle; pass --trusted-scenes only for trusted local assets')
        result = evaluate(request, backend=a.backend, batch_size=a.batch_size)
    result['request_sha256'] = hashlib.sha256(Path(a.request).read_bytes()).hexdigest()
    import numpy as np
    def serial(value):
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        raise TypeError(type(value).__name__)
    with Path(a.output).open('x') as f:
        json.dump(result, f, default=serial, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
