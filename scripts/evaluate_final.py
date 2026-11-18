"""Complete test inference and controlled cube directions, two GPU shards/model.

Immutable inputs, row-stable noise, bounded batches, explicit failure coverage.
Artifacts are NumPy arrays, atomic progress and audited summaries; no physics.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
BASE = Path(__file__).resolve().parents[2]
DTYPE = np.dtype([('finite','?'),('decoded','?'),('fit','?'),('direction_pass','?'),
                  ('mean_mm','f4'),('max_mm','f4'),('angle_deg','f4'),('ik_seconds','f4'),
                  ('q','f4',(22,)),('R','f4',(3,3)),('t','f4',(3,))])

def save(path, value):
    p=Path(path); tmp=p.with_suffix('.partial'); tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); tmp.replace(p)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def now(): return datetime.now(timezone.utc).isoformat()

def prepare(out):
    from dexgrasp.data.formal_corpus import verified_packed_identity
    from dexgrasp.train.runner import source_state
    from dexgrasp.test.directions import cube_directions
    from dexgrasp.test.formal import check_model_source
    import torch
    root=BASE/'data/sharpa_formal_20261111/local_cache'
    out.mkdir(parents=True,exist_ok=False)
    identity=verified_packed_identity(root,.2)
    rows=np.load(root/'test_rows.npy'); np.save(out/'test_rows.npy',rows)
    with sqlite3.connect(f'file:{BASE}/outputs/site_formal_20261111/viewer.sqlite?mode=ro',uri=True) as c:
        meta=json.loads(c.execute('SELECT value FROM metadata').fetchone()[0])
        assert meta['packed_sha256']==identity['packed_sha256']
        objs=c.execute("SELECT object_index,object_id FROM objects WHERE split='test' ORDER BY object_id").fetchall()
        full=c.execute("SELECT row_id,object_index FROM samples JOIN objects USING(object_index) WHERE split='test' ORDER BY row_id").fetchall()
    full=np.asarray(full,dtype=np.int64); assert np.array_equal(rows,full[:,0])
    lookup={k:i for i,(k,_) in enumerate(objs)}
    oi=np.array([lookup[k] for k in full[:,1]],dtype=np.int32); np.save(out/'object_indices.npy',oi)
    reps=np.array([rows[np.flatnonzero(oi==i)[0]] for i in range(len(objs))]); np.save(out/'direction_rows.npy',reps)
    directions=cube_directions(); jobs=[]
    for gi,g in enumerate('ABCD'):
        ckpt=BASE/f'checkpoints/recovery_20261113/{g}_fixed/step_200000.pt'
        state=torch.load(ckpt,map_location='cpu',weights_only=False)
        check_model_source(state)
        assert state['data_identity']==identity and state['step']==200000
        assert all(torch.isfinite(t).all() for t in state['model'].values() if t.is_floating_point())
        jobs.append(dict(group=g,checkpoint=str(ckpt),checkpoint_sha256=digest(ckpt),step=state['step'],gpus=[gi*2,gi*2+1],training_config=state['config']))
        del state
    plan=dict(schema='dexgrasp-final-full-v1',created_at=now(),root=str(root),data_identity=identity,source=source_state(),
        jobs=jobs,objects=[o for _,o in objs],test_count=len(rows),direction_count=len(objs)*26,baseline_count=len(objs),directions=directions,
        split='test',candidates=1,sampling_steps=20,seed=8401,precision='BF16 autocast; FP32 ODE accumulation',
        noise='test: CPU generator(seed + packed row); cube: same noise from seed + representative packed row across 26 commands plus original-label baseline; identical across models',
        direction_selection='lowest packed test row per object; all 407 test objects; one fixed observation per object',
        aggregation='equal conditions within object, then equal objects; failures remain in denominators; 2000 asset-group bootstrap draws, seed 7301',
        ik='max point L2 <=5 mm and legal joints; midpoint initialization, max_nfev=100; not physical success',
        timing='CUDA synchronized; encoder + Euler sampling only; excludes warmup, loading, transfer, IK',physical_success_evaluated=False,
        input_sha256={f:digest(out/f) for f in ['test_rows.npy','object_indices.npy','direction_rows.npy']})
    save(out/'plan.json',plan)
    print('Prepared',plan['test_count'],len(objs),flush=True)


def inputs(plan,out,mode,rank):
    root=Path(plan['root']); rows=np.load(out/('test_rows.npy' if mode=='test' else 'direction_rows.npy'))
    oi=np.load(out/'object_indices.npy') if mode=='test' else np.arange(len(rows))
    if mode=='test':
        index=np.arange(len(rows))[rank::2]; packed=rows[index]; objects=oi[index]; commands=np.full(len(index),-1)
    else:
        index=np.arange(len(rows)*27)[rank::2]; objects=index//27; packed=rows[objects]; commands=index%27
    dirs=np.array(np.load(root/'grasp_direction_cmd.npy',mmap_mode='r')[packed])
    if mode!='test':
        vectors=np.asarray([d['command_direction'] for d in plan['directions']],dtype=np.float32)
        mask=commands<26; dirs[mask]=vectors[commands[mask]]
    return index,packed,objects,commands,dirs


def infer(out,group,rank):
    import torch
    from dexgrasp.train.runner import load_model
    from dexgrasp.train.execution import configure_execution
    from dexgrasp.train.objectives import sample_keypoints
    torch.set_num_threads(4)
    plan=json.loads((out/'plan.json').read_text()); job=next(j for j in plan['jobs'] if j['group']==group)
    assert digest(job['checkpoint'])==job['checkpoint_sha256']
    device=f"cuda:{job['gpus'][rank]}"; torch.cuda.set_device(device)
    model,state=load_model(job['checkpoint'],torch.device(device)); configure_execution(model,dict(state['config'],compile_transformer=False)); del state
    root=Path(plan['root']); points=np.load(root/'object_points.npy',mmap_mode='r'); obs=np.load(root/'observation_index.npy',mmap_mode='r')
    folder=out/group/f'rank{rank}'; folder.mkdir(parents=True,exist_ok=True)
    progress=dict(group=group,rank=rank,gpu=job['gpus'][rank],pid=os.getpid(),stage='benchmark',generated=0,updated_at=now())
    def update(**kw): progress.update(**kw,updated_at=now()); save(folder/'progress.json',progress)
    def generate(packed,dirs):
        condition=dict(object_points=torch.from_numpy(np.array(points[obs[packed]])).to(device),grasp_direction_cmd=torch.from_numpy(dirs).to(device))
        z=torch.stack([torch.randn((21,3),generator=torch.Generator().manual_seed(plan['seed']+int(row))) for row in packed]).to(device)
        torch.cuda.synchronize(); tick=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16): result=sample_keypoints(model,condition,steps=20,noise=z)
        torch.cuda.synchronize(); elapsed=time.perf_counter()-tick
        return result.cpu().numpy()*.2,elapsed
    update()
    _,packed,_,_,dirs=inputs(plan,out,'test',rank)
    benchmarks=[]; best=None
    for b in (128,256,512,1024):
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            generate(packed[:b],dirs[:b])
            _,elapsed=generate(packed[:b],dirs[:b]); peak=torch.cuda.max_memory_allocated()
            entry=dict(batch=b,seconds=elapsed,grasps_per_second=b/elapsed,peak_gib=peak/2**30)
            benchmarks.append(entry); print('BENCH',group,rank,entry,flush=True)
            if best is None or entry['grasps_per_second']>best['grasps_per_second']:best=entry
            update(benchmarks=benchmarks)
            if peak>torch.cuda.get_device_properties(device).total_memory*.88:break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); benchmarks.append(dict(batch=b,error='CUDA OOM')); break
    if best is None:raise RuntimeError('No safe batch')
    batch=best['batch']; save(folder/'benchmark.json',benchmarks); update(batch_size=batch)
    # Cube coverage first so a complete direction report is available early.
    for mode in ('directions','test'):
        index,packed,objects,commands,dirs=inputs(plan,out,mode,rank)
        target=folder/mode; target.mkdir(exist_ok=False)
        np.savez(target/'index.npz',index=index,packed=packed,objects=objects,commands=commands,directions=dirs)
        result=np.lib.format.open_memmap(target/'keypoints.npy',mode='w+',dtype='float32',shape=(len(index),21,3))
        start=0; seconds=0.; failures=0; started=time.time()
        update(stage=mode,generated=0,total=len(index),mode_started=now())
        while start<len(index):
            n=min(batch,len(index)-start)
            try:
                pred,elapsed=generate(packed[start:start+n],dirs[start:start+n])
            except torch.cuda.OutOfMemoryError:
                if batch<=1:raise
                batch=max(1,batch//2); torch.cuda.empty_cache(); update(batch_size=batch); continue
            except FloatingPointError:
                pred=np.full((n,21,3),np.nan,dtype=np.float32); elapsed=0.
                for i in range(n):
                    try:
                        one,dt=generate(packed[start+i:start+i+1],dirs[start+i:start+i+1]); pred[i]=one[0]; elapsed+=dt
                    except FloatingPointError:failures+=1
            result[start:start+n]=pred; start+=n; seconds+=elapsed
            result.flush(); update(generated=start,total=len(index),sampling_seconds=seconds,wall_seconds=time.time()-started,failures=failures,batch_size=batch)
        save(target/'generation.json',dict(count=len(index),sampling_seconds=seconds,failures=failures,batch_size=batch,finished_at=now(),index_sha256=digest(target/'index.npz')))
    update(stage='inference_completed')


_DEC=None
_DECODE_CACHE={}

def init_decode(spec):
    global _DEC
    from dexgrasp.test.formal import official_hand
    _DEC=official_hand(spec)

def decode_chunk(job):
    from dexgrasp.test.metrics import decode_metrics
    path,start,end=job; p=Path(path)
    if path not in _DECODE_CACHE:
        _DECODE_CACHE.clear()
        with np.load(p/'index.npz') as index: dirs=index['directions']
        _DECODE_CACHE[path]=(np.load(p/'keypoints.npy'),dirs)
    k,d=_DECODE_CACHE[path]
    result=np.zeros(end-start,dtype=DTYPE)
    for i in range(start,end):
        r=decode_metrics(k[i],_DEC,d[i]); j=i-start
        for a,b in [('finite','finite_output'),('decoded','decoded'),('fit','fit_success'),('direction_pass','direction_pass')]:result[a][j]=r[b]
        for a,b in [('mean_mm','mean_keypoint_error_mm'),('max_mm','max_keypoint_error_mm'),('angle_deg','direction_error_deg'),('ik_seconds','ik_seconds')]: result[a][j]=r.get(b,np.nan)
        for key in ('q','R','t'):result[key][j]=r.get(key,np.nan)
    return start,result


def decode_all(out,workers):
    plan=json.loads((out/'plan.json').read_text()); paths=[]
    for mode in ('directions','test'):
        for g in 'ABCD':
            for rank in range(2):paths.append(out/g/f'rank{rank}'/mode)
    with ProcessPoolExecutor(workers,mp_context=mp.get_context('spawn'),initializer=init_decode,initargs=(str(Path(plan['root'])/'hand/hand_spec.json'),)) as pool:
        for p in paths:
            if (p/'decoded.done.json').exists():continue
            n=len(np.load(p/'keypoints.npy',mmap_mode='r')); target=np.zeros(n,dtype=DTYPE)
            tick=time.time(); done=0; last_update=0
            # Bounded submission: no 1.7 million result objects kept in RAM.
            jobs=((str(p),s,min(n,s+256)) for s in range(0,n,256))
            for start,value in pool.map(decode_chunk,jobs,chunksize=1):
                target[start:start+len(value)]=value; done+=len(value)
                if time.time()-last_update>=2 or done==n:
                    save(p/'decode_progress.json',dict(completed=done,total=n,workers=workers,seconds=time.time()-tick,updated_at=now()))
                    last_update=time.time()
            np.save(p/'decoded.npy',target); save(p/'decoded.done.json',dict(count=n,sha256=digest(p/'decoded.npy'),seconds=time.time()-tick))
            aggregate(out,p.parent.parent.name,p.name)


def aggregate(out,group,mode):
    plan=json.loads((out/'plan.json').read_text()); paths=[out/group/f'rank{r}'/mode for r in range(2)]
    if not all((p/'decoded.done.json').exists() for p in paths):return
    pieces=[np.load(p/'decoded.npy') for p in paths]; inds=[np.load(p/'index.npz') for p in paths]
    a=np.concatenate(pieces); objects=np.concatenate([x['objects'] for x in inds]); commands=np.concatenate([x['commands'] for x in inds])
    metrics={'finite_output_rate':'finite','decode_rate':'decoded','ik_fit_success_rate':'fit','direction_pass_rate_10deg':'direction_pass','mean_keypoint_error_mm':'mean_mm','mean_max_keypoint_error_mm':'max_mm','direction_angular_error_deg':'angle_deg'}
    split=json.loads((Path(plan['root'])/'split_protocol.json').read_text()); groups=[split['object_groups'][obj] for obj in plan['objects']]
    unique=sorted(set(groups)); bootstrap=np.random.default_rng(7301).integers(len(unique),size=(2000,len(unique)))
    def summary(mask):
        data=a[mask]; oi=objects[mask]; macro={}; per=[]
        for obj in range(len(plan['objects'])):
            part=data[oi==obj]
            if not len(part):continue
            values={k:float(np.nanmean(part[v])) if np.isfinite(part[v]).any() else None for k,v in metrics.items()}
            per.append(dict(object_id=plan['objects'][obj],asset_group=groups[obj],candidates=len(part),**values))
        for k,v in metrics.items():
            vals=np.array([r[k] if r[k] is not None else np.nan for r in per]); grouped=np.array([np.nanmean([r[k] for r in per if r['asset_group']==g and r[k] is not None]) if any(r['asset_group']==g and r[k] is not None for r in per) else np.nan for g in unique])
            samples=np.nanmean(grouped[bootstrap],axis=1)
            macro[k]=dict(mean=float(np.nanmean(vals)) if np.isfinite(vals).any() else None,ci95=np.nanquantile(samples,[.025,.975]).tolist() if np.isfinite(samples).any() else None,coverage=float(np.isfinite(data[v]).mean()))
        return dict(candidates=len(data),objects=len(per),macro=macro,per_object=per)
    main=summary(np.ones(len(a),bool) if mode=='test' else commands<26)
    main.update(group=group,mode=mode,physical_success_evaluated=False,plan_sha256=digest(out/'plan.json'),finished_at=now(),
        timing=dict(generation_ms_per_grasp=1000*sum(json.loads((p/'generation.json').read_text())['sampling_seconds'] for p in paths)/len(a),ik_mean_ms=float(np.mean(a['ik_seconds'])*1000)),
        artifacts=[dict(path=str(p),decoded_sha256=digest(p/'decoded.npy'),keypoints_sha256=digest(p/'keypoints.npy')) for p in paths])
    if mode=='directions':
        main['baseline']=summary(commands==26)
        main['by_direction']=[dict(**d,**{k:v for k,v in summary(commands==i).items() if k!='per_object'}) for i,d in enumerate(plan['directions'])]
    save(out/group/f'{mode}_summary.json',main)
    print('SUMMARY',group,mode,main['candidates'],flush=True)


def run(out,workers):
    plan=json.loads((out/'plan.json').read_text()); children=[]; logs=[]
    save(out/'status.json',dict(status='running',stage='inference',pid=os.getpid(),started_at=now()))
    for group in 'ABCD':
        for rank in range(2):
            log=(out/f'{group}_{rank}.log').open('x');logs.append(log)
            children.append(subprocess.Popen([sys.executable,__file__,'infer','--output',str(out),'--group',group,'--rank',str(rank)],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1'}))
    codes=[c.wait() for c in children]
    for f in logs:f.close()
    if any(codes):raise RuntimeError(f'Inference failed: {codes}')
    save(out/'status.json',dict(status='running',stage='IK',pid=os.getpid(),updated_at=now()))
    decode_all(out,workers)
    assert all((out/g/f'{m}_summary.json').exists() for g in 'ABCD' for m in ('test','directions'))
    save(out/'status.json',dict(status='completed',pid=os.getpid(),finished_at=now()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','infer','decode']);p.add_argument('--output',type=Path,required=True);p.add_argument('--group');p.add_argument('--rank',type=int);p.add_argument('--workers',type=int,default=96);args=p.parse_args();out=args.output.resolve()
    try:
        if args.action=='prepare':prepare(out)
        elif args.action=='run':run(out,args.workers)
        elif args.action=='infer':infer(out,args.group,args.rank)
        else:
            save(out/'status.json',dict(status='running',stage='IK',pid=os.getpid(),updated_at=now()))
            decode_all(out,args.workers)
            assert all((out/g/f'{m}_summary.json').exists() for g in 'ABCD' for m in ('test','directions'))
            save(out/'status.json',dict(status='completed',pid=os.getpid(),finished_at=now()))
    except Exception as e:
        if args.action in ('run','decode'):save(out/'status.json',dict(status='failed',error=str(e),updated_at=now()))
        raise
