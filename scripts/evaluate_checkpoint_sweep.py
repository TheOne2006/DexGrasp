"""Run an immutable checkpoint inventory with bounded GPU/CPU concurrency.

No training, checkpoint writes, deletion, or test-based checkpoint selection.
The status file is atomically replaced for the read-only display service.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dexgrasp.test.checkpoint_policy import exclusion


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def valid_result(folder, job, plan):
    folder = Path(folder)
    try:
        p = json.loads((folder/'protocol.json').read_text())
        s = json.loads((folder/'summary.json').read_text())
        return (p['checkpoint_sha256'] == job['checkpoint_sha256'] and
                p['selection_sha256'] == plan['selection_sha256'] and
                p['sampling_steps'] == 20 and p['seed'] == 8401 and p['candidates'] == 8 and
                p['batch_size'] == 8 and p['precision'] == 'BF16 autocast; FP32 ODE accumulation' and
                s['protocol_sha256'] == digest(folder/'protocol.json') and
                s['candidates'] == plan['candidates_per_checkpoint'] and
                all(p['source']['source_sha256'].get(k) == v for k, v in plan['metric_source'].items()) and
                (folder/'candidates.jsonl').is_file() and (folder/'noise.npy').is_file())
    except (OSError, ValueError, KeyError):
        return False


def prepare(checkpoints, selection, output, reuse_root):
    repo = Path(__file__).resolve().parents[1]
    checkpoints, selection, output, reuse_root = [Path(p).resolve() for p in (checkpoints,selection,output,reuse_root)]
    if output.exists():
        raise FileExistsError(output)
    selected = json.loads(selection.read_text())
    if selected['split'] != 'test':
        raise ValueError('sweep requires a frozen test selection')
    output.mkdir(parents=True)
    shutil.copyfile(selection, output/'selection.json')
    snapshot = output/'source'
    for p in sorted((repo/'dexgrasp').rglob('*.py')) + [repo/'scripts/evaluate_formal.py', Path(__file__).resolve()]:
        target = snapshot/p.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    source_keys = ['dexgrasp/test/metrics.py','dexgrasp/geometry/decode.py',
                   'dexgrasp/geometry/direction.py','dexgrasp/geometry/hand.py','dexgrasp/train/objectives.py']
    plan = dict(schema='dexgrasp-checkpoint-sweep-v1', created_at=now(), checkpoint_root=str(checkpoints),
                selection=str(output/'selection.json'), selection_sha256=digest(output/'selection.json'),
                candidates_per_checkpoint=len(selected['rows'])*8, objects=selected['objects'],
                source=str(snapshot), python=str(repo/'.venv/bin/python'),
                metric_source={k:digest(repo/k) for k in source_keys},
                snapshot_sha256={str(p.relative_to(snapshot)):digest(p) for p in snapshot.rglob('*.py')},
                checkpoint_policy='eligible saved steps only; exclude training instability by run policy; preserve weights; no selection on test',
                jobs=[], aliases=[], excluded=[])
    for directory in sorted(checkpoints.iterdir()):
        if not directory.is_dir() or not directory.name.startswith(('A_','B_')):
            continue
        group = directory.name.split('_')[0]
        for path in sorted(directory.glob('step_*.pt')):
            step = int(path.stem.split('_')[1]); name = f'{group}_{step:06d}'
            reason = exclusion(path, step)
            if reason:
                plan['excluded'].append(dict(name=name, checkpoint=str(path), **reason))
                continue
            stat = path.stat()
            job = dict(name=name, group=group, step=step, checkpoint=str(path),
                       checkpoint_sha256=digest(path), bytes=stat.st_size,
                       output=str(output/'runs'/name), status='pending')
            old = reuse_root/name
            if valid_result(old, job, plan):
                job.update(output=str(old), status='reused')
            plan['jobs'].append(job)
            print('Inventoried',name,job['status'],flush=True)
        last = directory/'last.pt'
        if last.exists():
            reason = exclusion(last)
            if reason:
                plan['excluded'].append(dict(checkpoint=str(last), **reason))
                continue
            match = next((j for j in plan['jobs'] if Path(j['checkpoint']).parent == directory
                          and last.samefile(j['checkpoint'])), None)
            if match is None:
                last_hash = digest(last)
                match = next((j for j in plan['jobs'] if j['checkpoint_sha256']==last_hash), None)
            if match is None:
                raise ValueError('last.pt has an unarchived state; preserve and inventory it explicitly before running')
            plan['aliases'].append(dict(path=str(last), same_as=match['name']))
    if not plan['jobs']:
        raise ValueError('no checkpoints found')
    plan['jobs'].sort(key=lambda j:(j['step'],j['group']))
    save(output/'plan.json',plan)
    print('Plan ready:',len(plan['jobs']),'checkpoints',flush=True)


def run(plan_path, gpus, workers):
    plan_path = Path(plan_path).resolve(); root = plan_path.parent
    if workers < 1 or not gpus or len(gpus)!=len(set(gpus)):
        raise ValueError('positive workers and distinct GPU indices required')
    lock = (root/'scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads(plan_path.read_text())
    for relative, expected in plan['snapshot_sha256'].items():
        if digest(Path(plan['source'])/relative) != expected:
            raise ValueError('frozen evaluation source changed')
    if digest(plan['selection']) != plan['selection_sha256']:
        raise ValueError('frozen selection changed')
    state = dict(schema=plan['schema'], status='running', pid=os.getpid(), started_at=now(),
                 plan_sha256=digest(plan_path), gpus=gpus, workers_per_run=workers, jobs=[])
    previous_path = root/'status.json'
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        for j in previous.get('jobs',[]):
            if j.get('status')=='running' and Path(f"/proc/{j.get('pid')}").exists():
                raise RuntimeError('previous evaluation worker still exists; refusing duplicate work')
    for job in plan['jobs']:
        j = dict(job)
        reason = exclusion(j['checkpoint'], j['step'])
        if reason:
            j.update(status='excluded', exclusion=reason)
            state['jobs'].append(j)
            continue
        if valid_result(j['output'], j, plan):
            j['status'] = 'reused' if j['status']=='reused' else 'completed'
        else:
            if j['status']=='reused':
                raise ValueError('previous result changed: '+j['name'])
            folder = Path(j['output'])
            if folder.exists():
                folder.rename(folder.with_name(folder.name+f'_incomplete_{time.time_ns()}'))
            j['status'] = 'pending'
        state['jobs'].append(j)
    (root/'logs').mkdir(exist_ok=True); (root/'runs').mkdir(exist_ok=True)
    active = {}
    def update():
        state['updated_at'] = now()
        state['completed'] = sum(j['status'] in ('completed','reused') for j in state['jobs'])
        state['failed'] = sum(j['status']=='failed' for j in state['jobs'])
        save(root/'status.json',state)
    update()
    while any(j['status']=='pending' for j in state['jobs']) or active:
        for gpu in gpus:
            if gpu in active:
                continue
            j = next((j for j in state['jobs'] if j['status']=='pending'), None)
            if j is None:
                break
            command = [plan['python'], str(Path(plan['source'])/'scripts/evaluate_formal.py'), 'run',
                       '--checkpoint',j['checkpoint'],'--selection',plan['selection'],'--output',j['output'],
                       '--device',f'cuda:{gpu}','--workers',str(workers)]
            log = (root/'logs'/f"{j['name']}.log").open('a')
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1',
                                          'PYTHONDONTWRITEBYTECODE':'1'})
            j.update(status='running',pid=child.pid,gpu=gpu,started_at=now(),command=command)
            active[gpu]=(child,j,log); print('START',j['name'],'GPU',gpu,flush=True)
        update()
        time.sleep(3)
        for gpu,(child,j,log) in list(active.items()):
            code = child.poll()
            if code is None:
                continue
            log.close()
            valid = code==0 and valid_result(j['output'],j,plan)
            j.update(status='completed' if valid else 'failed', exit_code=code, finished_at=now())
            if not valid:
                j['error'] = 'See run log; no failed candidates or runs silently discarded.'
            del active[gpu]; print('END',j['name'],j['status'],flush=True)
        update()
    if not state['failed']:
        eligible = [j for j in state['jobs'] if j['status']!='excluded']
        destination = root/('comparison_eligible' if len(eligible)!=len(state['jobs']) else 'comparison')
        if eligible and not destination.exists():
            subprocess.run([plan['python'],str(Path(plan['source'])/'scripts/evaluate_formal.py'),'compare',
                            '--runs',*[j['output'] for j in eligible], '--output',str(destination)],check=True)
        state['status']='completed'
    else:
        state['status']='completed_with_failures'
    state['finished_at']=now(); update()
    print('SWEEP',state['status'],state['completed'],'/',len(state['jobs']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('prepare')
    for key in ('checkpoints','selection','output','reuse-root'):
        q.add_argument('--'+key,required=True)
    q=sub.add_parser('run');q.add_argument('--plan-path',required=True)
    q.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3]);q.add_argument('--workers',type=int,default=12)
    args=vars(p.parse_args());action=args.pop('action')
    (prepare if action=='prepare' else run)(**args)
