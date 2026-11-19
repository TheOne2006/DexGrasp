"""Decode the union of selected candidates once and compare all ranking arms in physics."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

import replay_g7_endpoints as replay
from dexgrasp.test.frozen import digest, save

ARMS=('random','paper','released')


def run_scene(sid):
    dest=folder/'physics'/f'{sid}.json'
    if dest.exists():
        return
    from dexgrasp.test.metrics import decode_metrics
    results={}
    for g in groups:
        with np.load(folder/g/f'{sid}.npz') as z:
            selection=z['selected']
            unique=z['unique']
            xyz=z['keypoints']
            direction=z['direction']
            np.testing.assert_array_equal(unique,np.unique(selection))
            for arm in (1,2):
                np.testing.assert_array_equal(selection[arm],np.argsort(-z[ARMS[arm]],kind='stable')[:10])
        qpos={}
        for row,idx in enumerate(unique):
            q=np.full((3,29),np.nan)
            for stage,k in enumerate(xyz[row]):
                pose=decode_metrics(k,hand,direction)
                if pose['decoded'] and pose['joint_limits_valid']:
                    quat=Rotation.from_matrix(np.asarray(pose['R'])@frame.T).as_quat(scalar_first=True)
                    q[stage]=np.r_[pose['t'],quat,pose['q']]
            qpos[int(idx)]=q
        replay.poses={g:qpos}
        replay.groups=g
        measured=replay.public_scene(sid,list(qpos))
        results[g]=dict(selected=selection.tolist(),verdicts={str(i):measured[f'{g}/{i}'] for i in qpos})
    save(dest,results)


def summarize(output,ids,groups):
    plan=replay.read(replay.OLD/'official/plan.json')
    objects=np.array([plan['scenes'][i]['source']+'/'+plan['scenes'][i]['object'] for i in ids])
    results=[replay.read(output/'physics'/f'{i}.json') for i in ids]
    out={}
    for group in groups:
        out[group]={}
        for engine in ('bench','force'):
            table={}
            deltas={}
            for j,arm in enumerate(ARMS):
                rows=[r[group] for r in results]
                outcomes=np.array([[r['verdicts'][str(i)][engine] for i in r['selected'][j]] for r in rows])
                ok=outcomes.any(1)
                table[arm]=dict(passed=int(outcomes.sum()),total=outcomes.size,
                                scenes_passed=int(ok.sum()),scenes=len(ok),
                                objects_passed=len(np.unique(objects[ok])),objects=len(np.unique(objects)))
                v=np.stack((outcomes.mean(1),ok.astype(float)),axis=1)
                if arm=='random':
                    random=v
                else:
                    by_obj=np.stack([(v[objects==o]-random[objects==o]).mean(0) for o in np.unique(objects)])
                    rng=np.random.default_rng(78135)
                    ci=np.quantile(by_obj[rng.integers(len(by_obj),size=(4000,len(by_obj)))].mean(1),[.025,.975],axis=0)
                    deltas[arm]=dict(gsr_pp=float(by_obj[:,0].mean()*100),osr10_pp=float(by_obj[:,1].mean()*100),
                                     gsr_ci95_pp=(ci[:,0]*100).tolist(),osr10_ci95_pp=(ci[:,1]*100).tolist())
            out[group][engine]=dict(arms=table,paired_object_bootstrap=deltas)
    save(output/'summary.json',out)


def main(args):
    global folder,groups,hand,frame
    folder,groups=args.output,args.groups
    design=replay.read(folder/'design.json')
    ids=design['scene_ids']
    if args.smoke:
        ids=ids[:2]
    (folder/'physics').mkdir(exist_ok=True)
    sys.path.insert(0,str(replay.PUBLIC))
    sys.path.append('/opt/isaacsim/lib/python3.12/site-packages')
    import mujoco
    if mujoco.__version__!='3.3.3':
        raise RuntimeError('Wrong public benchmark environment')
    from dexonomy.op.gen_grasp import GraspFilter
    from dexgrasp.geometry.hand import URDFHand
    from dexgrasp.test.bench_force import load_backend,load_config
    from omegaconf import OmegaConf
    replay.public_plan=replay.read(replay.OLD/'official/plan.json')
    path=Path(replay.public_plan['hand_spec'])
    hand=URDFHand(path)
    replay.spec=hand.spec
    frame=np.asarray(hand.spec['official_from_grasp_rotation'])
    replay.settings=OmegaConf.load(replay.PUBLIC/'dexonomy/config/op/grasp.yaml')
    replay.backend,replay.bench_cfg=load_backend(replay.BENCH),load_config(replay.BENCH)
    replay.suite='official'
    meta=dict(design=digest(folder/'design.json'),scene_ids=ids,groups=groups,code=digest(__file__),
              executor=digest(replay.__file__),mujoco=mujoco.__version__,inputs={
                  g:[digest(folder/g/f'{i}.npz') for i in ids] for g in groups})
    protocol=folder/'physics_protocol.json'
    if protocol.exists() and replay.read(protocol)!=meta:
        raise ValueError('Ranking evaluation inputs changed')
    save(protocol,meta)
    with ProcessPoolExecutor(args.workers) as pool:
        for _ in pool.map(run_scene,ids):
            pass
    summarize(folder,ids,groups)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--groups',default='ABCD')
    p.add_argument('--workers',type=int,default=10)
    p.add_argument('--smoke',action='store_true')
    main(p.parse_args())
