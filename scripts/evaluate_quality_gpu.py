"""GPU static contact metrics and pooled pose PCA for frozen full-test outputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dexgrasp.test.frozen import decoded, digest, now, save

BASE = Path(__file__).resolve().parents[2]


def prepare(root):
    plan=json.loads((root/'plan.json').read_text());data=Path(plan['root']);out=root/'quality';out.mkdir(exist_ok=False)
    rows=np.load(root/'test_rows.npy');obs=np.load(data/'observation_index.npy')[rows];needed=set(obs.tolist());entries=[]
    manifests={};counter=0
    for name,identity in plan['data_identity']['source_identity'].items():
        package=BASE/'data/source_packages'/name;mf=package/'MANIFEST.json';assert digest(mf)==identity['manifest_sha256']
        files=json.loads(mf.read_text())['files'];manifests[name]={f['path']:f for f in files}
        for f in files:
            if f['path'].startswith('observations/') and f['path'].endswith('.npz'):
                if counter in needed:entries.append((counter,package,f))
                counter+=1
    points=np.load(data/'object_points.npy',mmap_mode='r')
    assert counter==len(points) and len(entries)==len(needed)
    def load(entry):
        i,package,f=entry;path=package/f['path'];assert digest(path)==f['sha256']
        with np.load(path,allow_pickle=False) as z:
            center=z['center_m'];scale=z['scale'];asset=str(z['asset']);obj=str(z['object_id'])
            np.testing.assert_allclose(points[i],(z['points_m']-center)/.2,atol=1e-6)
        asset_hash=manifests[package.name][asset]['sha256'];decomp=BASE/'data/eval_physics_sharpa_v1/objects'/asset_hash/'manifest.json'
        return dict(observation=i,object_id=obj,center=center.tolist(),scale=scale.tolist(),asset_sha256=asset_hash,decomposition=str(decomp))
    with ThreadPoolExecutor(32) as pool:metadata=list(pool.map(load,entries))
    scratch=Path(tempfile.mkdtemp(prefix='dexgrasp-quality-'));copy_jobs={}
    hand=BASE/'data/eval_physics_sharpa_v1/hand'
    for p in hand.rglob('*'):
        if p.is_file():copy_jobs[p]=(scratch/'hand'/p.relative_to(hand),None)
    groups={}
    for r in metadata:
        key=r['asset_sha256']+':'+','.join(map(str,r['scale']));group=groups.setdefault(key,dict(asset_sha256=r['asset_sha256'],scale=r['scale'],observations=[],pieces=[]))
        group['observations'].append(r['observation'])
        if not group['pieces']:
            m=json.loads(Path(r['decomposition']).read_text())
            for path,sha in m['pieces'].items():
                src=Path(r['decomposition']).parent/Path(path).name;dst=scratch/'objects'/r['asset_sha256']/src.name;copy_jobs[src]=(dst,sha);group['pieces'].append(str(dst))
    def copy(item):
        src,(dst,sha)=item
        if sha:assert digest(src)==sha
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    with ThreadPoolExecutor(32) as pool:list(pool.map(copy,copy_jobs.items()))
    centers=np.zeros((len(points),3),dtype=np.float32)
    for r in metadata:centers[r['observation']]=r['center']
    np.save(out/'centers.npy',centers);np.save(out/'test_observations.npy',obs)
    jobs=sorted(groups.values(),key=lambda g:(g['asset_sha256'],g['scale']))
    for i,g in enumerate(jobs):g['id']=i
    protocol=dict(schema='dexgrasp-static-quality-v1',created_at=now(),parent_plan_sha256=digest(root/'plan.json'),
        test_count=len(rows),objects=len(plan['objects']),observations=len(needed),groups=jobs,
        hand_xml=str(scratch/'hand/collision.xml'),hand_spec=str(data/'hand/hand_spec.json'),
        source_sha256=digest(__file__),metric_source_sha256=digest(BASE/'DexGrasp/dexgrasp/test/physics.py'),
        geometry='MuJoCo Warp static collision queries; official Sharpa collision geoms, cached CoACD object pieces; 10 mm contact query margin',
        cdc='range of five finger minimum signed distances; missing finger distance capped at +10 mm',
        pd='maximum hand-object contact penetration per grasp, metres converted to mm',
        diversity='PC1 variance percentage of pooled unstandardized [world/object-origin t(m), official wrist rotvec(rad), q(rad)]; all finite decoded test poses; no physical-success filter',
        aggregation='geometry: object-balanced; PCA: all-candidate pooled covariance',
        gsr=None,osr=None,physical_success_evaluated=False,paper_comparable=False,
        source_note='KPGrasp Table 1/3 are published Shadow/Dexonomy results; these are Sharpa adapted diagnostics, without likelihood ranking',
        observation_metadata=metadata,staged_asset_files=len(copy_jobs))
    save(out/'protocol.json',protocol);print('Prepared',len(jobs),'geometry groups',len(needed),'observations',flush=True)


def diversity(root):
    from scipy.spatial.transform import Rotation
    from dexgrasp.test.physics import pca_diversity
    out=root/'quality';p=json.loads((out/'protocol.json').read_text());centers=np.load(out/'centers.npy')[np.load(out/'test_observations.npy')]
    hand=json.loads(Path(p['hand_spec']).read_text());frame=np.asarray(hand['official_from_grasp_rotation'])
    results={}
    for g in 'ABCD':
        a=decoded(root,g);valid=a['decoded'];rot=a['R'][valid]@frame.T
        features=np.column_stack((a['t'][valid]+centers[valid],Rotation.from_matrix(rot).as_rotvec(),a['q'][valid]))
        x=features-features.mean(0);ev=np.linalg.eigvalsh(x.T@x);ev=np.maximum(ev,0)[::-1]
        results[g]=dict(pc1_pct=float(100*ev[0]/ev.sum()),cumulative_pct=(100*np.cumsum(ev[:5])/ev.sum()).tolist(),candidates=len(features),coverage=float(valid.mean()))
    save(out/'diversity.json',dict(protocol_sha256=digest(out/'protocol.json'),models=results));print('PCA',results,flush=True)


def worker(root,gpu,limit=None):
    import gc
    import mujoco
    import mujoco_warp as mjw
    import warp as wp
    from scipy.spatial.transform import Rotation
    from dexgrasp.geometry.hand import URDFHand
    from dexgrasp.test.physics import PoseEvaluator,FINGERS
    from quality_kernels import reduce_contacts
    sys.path.insert(0,str(BASE/'DexGrasp/simulation'))
    from dexonomy.sim.mujoco_warp_env import check_warp_allocation
    wp.set_device(f'cuda:{gpu}');out=root/'quality';p=json.loads((out/'protocol.json').read_text());obs=np.load(out/'test_observations.npy');centers=np.load(out/'centers.npy')
    arrays={g:decoded(root,g) for g in 'ABCD'};hand=URDFHand(p['hand_spec']);groups=p['groups'][gpu::8];groups=groups if limit is None else groups[:limit]
    folder=out/f'gpu{gpu}';folder.mkdir(exist_ok=True);total=0;started=time.time()
    for gi,group in enumerate(groups):
        path=folder/f"{group['id']:05d}.npz"
        if path.exists():continue
        indices=np.flatnonzero(np.isin(obs,group['observations']));n=len(indices)
        e=PoseEvaluator(hand,p['hand_xml'],group['pieces'],group['scale']);e.model.geom_margin[e.object_geoms]=.01
        # Static collision only: no solver or stepping is invoked.
        e.model.opt.noslip_iterations=0
        original_flags=e.model.opt.disableflags
        e.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        model=mjw.put_model(e.model)
        e.model.opt.disableflags=original_flags
        categories=np.full(e.model.ngeom,-1,np.int32)
        for geom in range(e.model.ngeom):
            name=e.names[e.model.geom_bodyid[geom]]
            if name=='object':categories[geom]=5
            else:
                for i,f in enumerate(FINGERS):
                    if name.startswith('right_'+f+'_'):categories[geom]=i
        cats=wp.array(categories,dtype=int)
        all_R=np.concatenate([a['R'][indices] for a in arrays.values()]);all_t=np.concatenate([a['t'][indices]+centers[obs[indices]] for a in arrays.values()]);all_q=np.concatenate([a['q'][indices] for a in arrays.values()]);valid=np.concatenate([a['decoded'][indices] for a in arrays.values()])
        measured=np.full((4*n,2),np.nan,np.float32);batch=512
        data=mjw.make_data(e.model,nworld=batch,nconmax=512,nccdmax=128,njmax=32)
        validation=[];cpu_fallback=False
        for start in range(0,4*n,batch):
            end=min(start+batch,4*n);ids=np.arange(start,end);ids=ids[valid[ids]]
            if not len(ids):continue
            padding=np.pad(ids,(0,batch-len(ids)),mode='edge');qpos=np.tile(e.model.qpos0,(batch,1)).astype(np.float32);qpos[:,e.qadr]=all_q[padding]
            data.qpos.assign(qpos);data.mocap_pos.assign(all_t[padding,None,:].astype(np.float32));quat=Rotation.from_matrix(all_R[padding]@e.official_from_grasp.T).as_quat(scalar_first=True).astype(np.float32);data.mocap_quat.assign(quat[:,None,:]);data.overflow.zero_()
            mjw.kinematics(model,data);mjw.com_pos(model,data);mjw.collision(model,data)
            wp.synchronize();check_warp_allocation(data,mjw.OverflowType,context='static quality')
            minima=wp.full((batch,5),.01,dtype=float);pd=wp.zeros(batch,dtype=float)
            wp.launch(reduce_contacts,data.naconmax,inputs=[data.nacon,data.contact.worldid,data.contact.geom,data.contact.dist,cats,minima,pd])
            distances=minima.numpy()[:len(ids)];values=np.column_stack(((distances.max(1)-distances.min(1))*1000,pd.numpy()[:len(ids)]*1000));measured[ids]=values
            if start==0:
                for j in np.unique(np.linspace(0,len(ids)-1,min(8,len(ids)),dtype=int)):
                    i=ids[j];cpu=e.evaluate(all_R[i],all_t[i],all_q[i],simulate=False);expected=np.array([cpu['contact_distance_consistency_mm'],cpu['hand_object_penetration_mm']]);err=np.abs(values[j]-expected)
                    validation.append(dict(index=int(indices[i%n]),model='ABCD'[i//n],gpu=values[j].tolist(),cpu=expected.tolist(),max_error_mm=float(err.max())))
                # Different narrow-phase implementations must agree before full publication.
                if max(r['max_error_mm'] for r in validation)>.2:cpu_fallback=True
                save(folder/f"{group['id']:05d}.validation.json",validation)
        if cpu_fallback:
            for i in np.flatnonzero(valid):
                cpu=e.evaluate(all_R[i],all_t[i],all_q[i],simulate=False)
                measured[i]=[cpu['contact_distance_consistency_mm'],cpu['hand_object_penetration_mm']]
        np.savez(path,indices=indices,cdc_mm=measured[:,0].reshape(4,n),pd_mm=measured[:,1].reshape(4,n),cpu_fallback=np.array(cpu_fallback))
        total+=4*n;save(folder/'progress.json',dict(status='running',gpu=gpu,groups_completed=gi+1,groups_total=len(groups),candidates=total,cpu_fallback_last_group=cpu_fallback,seconds=time.time()-started,updated_at=now()))
        del data,model,e;gc.collect()
    save(folder/'status.json',dict(status='completed',groups=len(groups),candidates=total,seconds=time.time()-started))


def summarize(root):
    out=root/'quality';p=json.loads((out/'protocol.json').read_text());n=p['test_count'];oi=np.load(root/'object_indices.npy');models={};a=np.full((4,n,2),np.nan,np.float32);covered=np.zeros(n,bool)
    for f in out.glob('gpu*/*.npz'):
        with np.load(f) as z:
            ids=z['indices'];assert not covered[ids].any();covered[ids]=True;a[:,ids,0]=z['cdc_mm'];a[:,ids,1]=z['pd_mm']
    assert covered.all(),int(covered.sum())
    div=json.loads((out/'diversity.json').read_text())['models']
    for i,g in enumerate('ABCD'):
        values=a[i];per=[np.nanmean(values[oi==k],axis=0) for k in range(p['objects'])]
        mean=np.nanmean(per,axis=0);models[g]=dict(cdc_mm=float(mean[0]),pd_mm=float(mean[1]),diversity_pct=div[g]['pc1_pct'],candidates=n,geometry_coverage=float(np.isfinite(values).all(1).mean()),gsr_pct=None,osr_pct=None)
    save(out/'summary.json',dict(protocol_sha256=digest(out/'protocol.json'),models=models,physical_success_evaluated=False,paper_comparable=False));save(out/'status.json',dict(status='completed',finished_at=now()))


def run(root):
    out=root/'quality';save(out/'status.json',dict(status='running',pid=os.getpid(),started_at=now()));children=[]
    for gpu in range(8):
        log=(out/f'gpu{gpu}.log').open('a');children.append(subprocess.Popen([sys.executable,__file__,'worker','--root',str(root),'--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}))
    codes=[c.wait() for c in children]
    if any(codes):raise RuntimeError(f'Geometry workers failed {codes}')
    summarize(root)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','diversity','worker','run','summarize']);p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',type=int,default=0);p.add_argument('--limit',type=int);args=p.parse_args();root=args.root.resolve()
    try:
        if args.action=='worker':worker(root,args.gpu,args.limit)
        else:globals()[args.action](root)
    except Exception as e:
        if args.action=='run':save(root/'quality/status.json',dict(status='failed',error=str(e),updated_at=now()))
        raise
