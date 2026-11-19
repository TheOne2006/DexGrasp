"""Replay saved three-stage hand commands with the historical native and public executors."""
import argparse
import fcntl
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
BASE = REPO.parent
OLD = BASE / 'outputs/gsr_osr_200k'
HAND = BASE / 'data/simulation_assets/hand/sharpa_v2/right_kp.xml'
PUBLIC = BASE / 'reference/Dexonomy'
BENCH = BASE / 'reference/DexGraspBench'
sys.path.insert(0, str(REPO))
from dexgrasp.test.frozen import digest, save


def read(path):
    return json.loads(path.read_text())


def public_scene(sid, ids):
    from dexgrasp.test.dexonomy_control import floating_scene, make_scene
    from dexgrasp.test.bench_force import build_replay, evaluate
    from dexonomy.util.file_util import load_scene_cfg
    from dexonomy.util.traj_util import get_planner
    if suite == 'official':
        relative = Path(public_plan['scenes'][sid]['path']).relative_to(public_plan['assets_root'])
        scene = load_scene_cfg(str(BASE / 'data/dexonomy_official_test' / relative))
    else:
        geometry = stage_plan['groups'][sid]
        scene = floating_scene(OLD / 'internal_stages/assets', geometry)
    scene, opt, env, target, order = make_scene(scene, HAND, spec, settings)
    obj = scene['scene'][scene['task']['obj_name']]
    bench = None
    if suite == 'official':
        bench = build_replay(backend, Path(obj['info_path']).parents[1], obj['scale'][0],
                             obj['density'], HAND, bench_cfg, obj['pose'])
    out = {}
    for group in groups:
        for i in ids:
            raw = poses[group][i]
            valid = np.isfinite(raw).all()
            row = dict(valid=bool(valid), force=False)
            if bench:
                row['bench'] = False
            if valid:
                q = raw.copy()
                q[:,7:] = q[:,7:][:,order]
                traj = get_planner(scene['task']).plan_trajectory(target.copy(), *q, None)
                passed, _, _ = env.eval_fc(*traj, .05, 15.)
                row['force'] = bool(passed and not env._data.warning.number.any() and np.isfinite(env._data.qpos).all())
                if suite == 'paired':
                    for name, arm in [('no_squeeze',q[[0,1,1]]),('hold',q[[1,1,1]])]:
                        control = get_planner(scene['task']).plan_trajectory(target.copy(), *arm, None)
                        ok, _, _ = env.eval_fc(*control,.05,15.)
                        row[name] = bool(ok and not env._data.warning.number.any() and np.isfinite(env._data.qpos).all())
                if bench:
                    row['bench'] = evaluate(bench, *q)
                if suite == 'noise' and row['force']:
                    import mujoco
                    env.reset_qpos(traj[0][0])
                    env.run_trajectory(*traj[:3])
                    mujoco.mj_forward(env._model, env._data)
                    body = env._obj_active_bodyid
                    points = defaultdict(list)
                    for contact in env._data.contact:
                        pair = [env._model.geom_bodyid[k] for k in (contact.geom1,contact.geom2)]
                        if body in pair and contact.dist <= .002:
                            link = env._model.body(pair[1-pair.index(body)]).name
                            if link.startswith(env._cfg.hand_prefix):
                                points[link].append((contact.pos-env._data.xpos[body]) @ env._data.xmat[body].reshape(3,3))
                    row['links'] = sorted(points)
                    row['centroid'] = np.mean([np.mean(v,axis=0) for v in points.values()],axis=0).tolist() if points else None
            out[f'{group}/{i}'] = row
    return out


def native_scene(sid, ids):
    from dexonomy.tabletop_protocol import evaluate, make_host
    from dexonomy.op.eval_force_closure_warp_mixed import evaluate_force_closure_data_warp_mixed
    path = source_dir / 'scenes' / f'{sid}.npy'
    scene = np.load(path, allow_pickle=True).item()
    obj = scene['scene'][scene['task']['obj_name']]['pose']
    world = Rotation.from_quat(obj[3:],scalar_first=True)
    entries, out = [], {}
    for g in groups:
        for i in ids:
            raw = poses[g][i].astype(float)
            name = f'{g}/{i}'
            valid = np.isfinite(raw).all()
            out[name] = dict(valid=bool(valid), force=False)
            if valid:
                raw[:,:3] = world.apply(raw[:,:3]) + obj[:3]
                raw[:,3:7] = (world * Rotation.from_quat(raw[:,3:7],scalar_first=True)).as_quat(scalar_first=True)
                raw[:,7:] = raw[:,7:][:,native_order]
                entries.append(dict(id=name,data=dict(pipeline_stage='stability_candidate',scene_path=str(path),
                                    pregrasp_qpos=raw[:1],grasp_qpos=raw[1:2],squeeze_qpos=raw[2:])))
    if entries and scene['task']['type'] == 'slide':
        result = evaluate(entries,[make_host(str(path),str(HAND))],[0]*len(entries),
                          backend='warp',batch_size=min(30,len(entries)*6))
        passed = {r['candidate_id']:bool(r['passes']) for r in result['combined']}
    elif entries:
        result = evaluate_force_closure_data_warp_mixed([x['data'] for x in entries],cfg,
                  source_names=[x['id'] for x in entries],world_batch_size=min(90,len(entries)*6))
        passed = {}
        for r in result['results']:
            metrics = r.get('metrics') or {}
            if r.get('simulation_error') or metrics.get('simulation_error') or 'error' in metrics:
                raise RuntimeError(f'Physics error in {sid}')
            passed[r['source_name']] = bool(r['success'])
    else:
        passed = {}
    if set(passed) != {x['id'] for x in entries}:
        raise RuntimeError('Incomplete physical replay')
    for name,status in passed.items():
        out[name]['force'] = status
    return out


def run_scene(job):
    sid, ids = job
    path = chunks / f'{sid}.json'
    if not path.exists():
        tick = time.perf_counter()
        result = native_scene(sid, ids) if native else public_scene(sid, ids)
        save(path, dict(seconds=time.perf_counter()-tick, rows=result))
    return sid


def run_official_scene(job):
    # A second CPU pool can take the opposite end of the official scene list.
    # Locks are released by the OS on interruption. Both workers recheck the published receipt.
    sid, _ = job
    if (chunks / f'{sid}.json').exists():
        return sid
    with (chunks.parent / 'locks' / str(sid % 512)).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return run_scene(job)


def main(args):
    global groups,suite,native,public_plan,stage_plan,spec,settings,backend,bench_cfg
    global asset_dir,cfg,source_dir,native_order,poses,chunks
    groups,suite,native = args.groups,args.suite,args.native
    sys.path.insert(0,str(REPO/'simulation' if native else PUBLIC))
    sys.path.append('/opt/isaacsim/lib/python3.12/site-packages')
    from omegaconf import OmegaConf
    import mujoco
    expected = '3.17.0' if native else '3.3.3'
    if mujoco.__version__ != expected:
        raise RuntimeError(f'Expected MuJoCo {expected}')
    stage_plan = read(OLD/'internal_stages/plan.json')
    spec = read(Path(stage_plan['hand_spec']))
    dest = args.output/suite/('native' if native else 'public')
    chunks = dest/'scenes'
    chunks.mkdir(exist_ok=True,parents=True)
    poses, meta, hashes = {},{},{}
    for g in groups:
        poses[g] = {}
        for rank in (0,1):
            path = args.output/suite/g/f'rank{rank}'
            receipt = read(path/'decode.json')
            assert digest(path/'qpos.npy') == receipt['sha256']
            hashes[f'{g}/{rank}'] = receipt['sha256']
            with np.load(path/'index.npz') as z:
                info = dict(z)
            q = np.load(path/'qpos.npy')
            for j,idx in enumerate(info['index']):
                i = int(idx)
                poses[g][i] = q[j]
                meta[i] = int(info['packed'][j]) if 'packed' in info else None
        if set(poses[g]) != set(poses[groups[0]]):
            raise ValueError('Model draw mismatch')
    scenes = defaultdict(list)
    if native:
        source_dir = OLD/('source_tasks_paired' if suite=='paired' else 'source_conditioned')
        plan = read(source_dir/'plan.json')
        offset = 0
        if suite != 'paired':
            key = {'0.5':'half','1.5':'high'}.get(suite,suite)
            offset = plan['sets'][key]['start']
        for sid,rows in plan['sources'].items():
            for r in rows:
                i = r['index'] - offset
                if i in meta:
                    scenes[int(sid)].append(i)
        names = [n.attrib['name'] for n in ET.parse(HAND).findall('.//worldbody//joint') if n.get('name') in spec['joint_order']]
        assert sorted(names) == sorted(spec['joint_order'])
        native_order = [spec['joint_order'].index(n) for n in names]
        cfg = OmegaConf.create(dict(hand=dict(xml_path=str(HAND)),obj_mass=.1,physics_profile='native',
                warp_device='cuda:0',op=dict(miu_coef=[.6,.02],hard=False,noslip_iterations=0,
                skip_pregrasp=False,trans_thre=.05,rot_thre=15)))
    else:
        from dexonomy.op.gen_grasp import GraspFilter
        from dexgrasp.test.bench_force import load_backend,load_config
        settings = OmegaConf.load(PUBLIC/'dexonomy/config/op/grasp.yaml')
        if suite=='official':
            public_plan = read(OLD/'official/plan.json')
            backend,bench_cfg = load_backend(BENCH),load_config(BENCH)
            for i in meta:
                scenes[i//10].append(i)
        else:
            mapping = {o:g['id'] for g in stage_plan['groups'] for o in g['observations']}
            observations = np.load(BASE/'data/sharpa_formal_20261111/local_cache/observation_index.npy',mmap_mode='r')
            for i,packed in meta.items():
                scenes[mapping[int(observations[packed])]].append(i)
    if sorted(i for ids in scenes.values() for i in ids) != sorted(meta):
        raise ValueError('Incomplete scene mapping')
    protocol = dict(code=digest(__file__),qpos=hashes,hand=digest(HAND),executor=expected,
                    groups=groups,candidates=len(meta),scenes=len(scenes),world_size=args.world_size,
                    control='G7 pregrasp and squeeze endpoints; archived ABCD grasp')
    protocol_path = dest/f'protocol-{args.rank}.json'
    if protocol_path.exists() and read(protocol_path)!=protocol:
        raise RuntimeError('Resume protocol changed')
    save(protocol_path,protocol)
    jobs = sorted(scenes.items())[args.rank::args.world_size]
    if native:
        for job in jobs:
            run_scene(job)
    else:
        worker = run_scene
        if suite == 'official':
            (chunks.parent / 'locks').mkdir(exist_ok=True)
            worker = run_official_scene
        if args.reverse:
            jobs.reverse()
        with ProcessPoolExecutor(args.workers) as pool:
            for _ in pool.map(worker,jobs,chunksize=1):
                pass


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--suite',required=True)
    p.add_argument('--native',action='store_true')
    p.add_argument('--reverse',action='store_true',help='Claim public scenes in reverse order')
    p.add_argument('--rank',type=int,default=0)
    p.add_argument('--world-size',type=int,default=1)
    p.add_argument('--workers',type=int,default=24)
    p.add_argument('--groups',default='ABCD')
    main(p.parse_args())
