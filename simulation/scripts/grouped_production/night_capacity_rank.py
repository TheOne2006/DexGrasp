"""One rank owner: bounded lower-capacity attempts, each with a fresh guarded process."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,time,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from night_capacity_policy import capacity_failure,lower,selected,adaptive,initialize,adapt_failure,systemic_reason
from night_checkpoint_identity import token

def save_json(path,value):
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)

def save_state(path,state):
    tmp=path.with_suffix('.partial')
    with tmp.open('wb') as f:np.save(f,state,allow_pickle=True);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def locate(attempt,checkpoint):
    events=[]
    for f in (attempt/'audit').glob('events-*.jsonl'):
        for line in f.read_text().splitlines():
            try:events.append(json.loads(line))
            except ValueError:continue
    events.sort(key=lambda x:x.get('time',0));job=None;stage=None
    for e in events:
        if e['kind']=='vnext_job_start':job=e;stage=None
        elif e['kind']=='refine_batch_start':stage='refine'
        elif e['kind']=='eval_batch_start':stage='eval'
        elif e['kind']=='night_stage_start' and e.get('stage') in ('refine','eval'):stage=e['stage']
    if job is None or stage is None:raise RuntimeError('Capacity failure stage is not localized')
    folder=checkpoint/'jobs'/job['tag'];input_file=folder/('init.npy' if stage=='refine' else 'refine.npy')
    if not input_file.exists() or (folder/(stage+'.npy')).exists():raise RuntimeError('Capacity stage transaction is not uncommitted')
    return job,stage,input_file

def main():
    p=argparse.ArgumentParser();p.add_argument('config',type=Path);p.add_argument('--name',default='night');a=p.parse_args();c=json.loads(a.config.read_text());root=Path(c['output_root']);root.mkdir(exist_ok=True)
    owner=(root/'capacity-owner.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save_json(root/'capacity-process.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split(')')[-1].split()[19]))
    if not adaptive(c):assert c['capacity_policy']['max_failures_per_rank']>0
    script=Path(__file__).parent;statefile=Path(c['checkpoint_root'])/'state.npy';stop=root/'stop-request.json'
    attempts=json.loads((root/'capacity-attempts.json').read_text()) if (root/'capacity-attempts.json').exists() else []
    previous=root/'capacity-result.json'
    if previous.exists():
        history=root/'capacity-supervisor-history';history.mkdir(exist_ok=True)
        os.rename(previous,history/f'{time.time_ns()}-result.json')
    signal.signal(signal.SIGTERM,lambda *_:save_json(stop,dict(reason='capacity supervisor signal',hard_deadline_epoch=time.time()+600)))
    signal.signal(signal.SIGINT,lambda *_:save_json(stop,dict(reason='capacity supervisor signal',hard_deadline_epoch=time.time()+600)))
    while True:
        if stop.exists():break
        if adaptive(c) and statefile.exists():
            current=np.load(statefile,allow_pickle=True).item();initialize(c,current)
            if systemic_reason(c,current):raise RuntimeError('Systemic capacity protection: '+systemic_reason(c,current))
            del current
        number=1
        while (root/'runs'/f'{a.name}-{number:02d}').exists():number+=1
        name=f'{a.name}-{number:02d}';attempt=root/'runs'/name
        save_json(root/'active-attempt.json',dict(name=name,at=time.time(),state='starting'))
        child=subprocess.Popen([c['python'],'-B',str(script/'run.py'),str(a.config),'--name',name])
        while child.poll() is None:
            if stop.exists() and attempt.exists() and not (attempt/'stop-request.json').exists():save_json(attempt/'stop-request.json',json.loads(stop.read_text()))
            time.sleep(1)
        if not (attempt/'result.json').exists():raise RuntimeError('Guard did not produce a terminal result; no retry')
        result=json.loads((attempt/'result.json').read_text());attempts.append(dict(name=name,**result));save_json(root/'capacity-attempts.json',attempts)
        if result['status']=='PASS' or stop.exists():break
        rows=[]
        for line in (attempt/'resources.jsonl').read_text().splitlines():
            try:rows.append(json.loads(line))
            except ValueError:continue
        console=(attempt/'console.log').read_text(errors='replace')
        if not capacity_failure(result,rows,console):raise RuntimeError('Non-capacity failure: no retry')
        # All worker state must be gone. The next guard reacquires every GPU/rank lock.
        identity=json.loads((attempt/'process.json').read_text());proc=Path(f"/proc/{identity['pid']}/stat")
        if proc.exists() and proc.read_text().split(')')[-1].split()[19]==str(identity['start_ticks']):raise RuntimeError('Old worker remains alive')
        if int(result['gpu_after'])>=256:raise RuntimeError('GPU memory not released')
        query=['nvidia-smi','-i',c['gpu_uuid'],'--query-compute-apps=pid','--format=csv,noheader,nounits']
        if subprocess.check_output(query,text=True).strip():raise RuntimeError('GPU owner remains or changed; no retry')
        state=np.load(statefile,allow_pickle=True).item();assert state['config_hash']==token(c,script/'night_worker.py')
        job,stage,input_file=locate(attempt,statefile.parent)
        expected=state['artifacts'].get(str(input_file.relative_to(statefile.parent)));actual=hashlib.sha256(input_file.read_bytes()).hexdigest()
        if expected!=actual:raise RuntimeError('Capacity replay input changed or uncommitted')
        if adaptive(c):
            frozen=np.load(input_file,allow_pickle=True).item()
            ids=[entry['id'] for entry in frozen['entries']]
            if len(ids)!=len(set(ids)):raise RuntimeError('Duplicate capacity input IDs')
            record=adapt_failure(c,state,job['group'],job['family'],stage,tag=job['tag'],attempt=str(attempt),reason=result.get('stop_reason') or 'CUDA_OOM',input_sha256=actual,input_path=str(input_file.relative_to(statefile.parent)),candidate_ids=ids)
            del frozen
        else:
            record=lower(c,state,job['group'],job['family'],stage,attempt=str(attempt),reason=result.get('stop_reason') or 'CUDA_OOM',input_sha256=actual)
        save_state(statefile,state);save_json(root/'capacity-ledger.json',state['capacity_history']);save_json(root/'active-attempt.json',dict(name=name,at=time.time(),state='rebuilding',capacity_change=record));print(json.dumps(record),flush=True)
        if adaptive(c):
            save_json(root/'capacity-deferred.json',state.get('capacity_deferred',{}))
            if systemic_reason(c,state):raise RuntimeError('Systemic capacity protection: '+systemic_reason(c,state))
        del state
    save_json(root/'capacity-result.json',dict(status='PASS',attempts=len(attempts),stopped=stop.exists(),at=time.time()))

if __name__=='__main__':
    try:main()
    except Exception as error:
        import sys
        cfg=json.loads(Path(sys.argv[1]).read_text());save_json(Path(cfg['output_root'])/'capacity-result.json',dict(status='FAIL',error=repr(error),at=time.time()))
        raise
