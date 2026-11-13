"""Durable two-lane experiment supervisor. A->C and B->D advance independently.

A failed run never starts its successor. Transient exits resume only from an
atomic checkpoint, with bounded retries; state and process records are atomic.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def timestamp():return datetime.now(timezone.utc).isoformat()


def checkpoint_step(path):
    if not path.exists():return None
    import torch
    return int(torch.load(path,map_location='cpu',weights_only=False,mmap=True)['step'])


def numerical_failure(out, log_path, start_offset=0):
    """Numerical failures require diagnosis; retrying last.pt can repeat damage."""
    if (out/'numerical_failure.json').exists():
        return True
    if not log_path.exists():
        return False
    with log_path.open('rb') as stream:
        stream.seek(max(start_offset, log_path.stat().st_size-131072))
        tail = stream.read().decode(errors='replace').lower()
    return any(marker in tail for marker in ('floatingpointerror', 'non-finite', 'nonfinite'))


def run_lane(lane, plan, root):
    state_path=root/(lane['name']+'.json')
    env={**os.environ,**plan['environment'],'CUDA_VISIBLE_DEVICES':','.join(map(str,lane['gpus']))}
    env.pop('DEXGRASP_ISOLATED_GPU',None)
    for run in lane['runs']:
        config=Path(run['config']);cfg=json.loads(config.read_text());out=Path(cfg['run_dir']);ckpt=Path(cfg['checkpoint_dir'])/'last.pt'
        record=out.parent/(out.name+'_process.json')
        previous=json.loads(record.read_text()) if record.exists() else {}
        previous_pid=previous.get('pid')
        if previous_pid and Path(f'/proc/{previous_pid}/cmdline').exists():
            command_line=Path(f'/proc/{previous_pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if str(config) in command_line:
                raise RuntimeError(f'Existing worker {previous_pid} is still running; refusing duplicate launch')
        if (out/'numerical_failure.json').exists():
            write_json(state_path,dict(lane=lane['name'],run=run['id'],status='failed',
                reason='recorded numerical failure requires a separately diagnosed recovery run',at=timestamp()))
            return False
        for attempt in range(4):
            saved=checkpoint_step(ckpt)
            if saved==cfg['max_steps']:
                write_json(state_path,dict(lane=lane['name'],run=run['id'],status='completed',step=saved,at=timestamp()))
                break
            command=[plan['python'],'-m','torch.distributed.run','--standalone',f"--nproc_per_node={len(lane['gpus'])}",
                     str(Path(plan['source_root'])/'scripts/train_fm_ddp.py'),'--config',str(config)]
            if saved is not None:command+=['--resume',str(ckpt)]
            elif out.exists() and any(out.iterdir()):
                # No weight update may be represented as a restart without a checkpoint.
                write_json(state_path,dict(lane=lane['name'],run=run['id'],status='failed',reason='existing output without checkpoint',at=timestamp()))
                return False
            log_path=root/(run['id']+'.log')
            log_offset=log_path.stat().st_size if log_path.exists() else 0
            with log_path.open('ab',buffering=0) as log:
                process=subprocess.Popen(command,cwd=plan['source_root'],env=env,stdout=log,stderr=subprocess.STDOUT)
                info=dict(pid=process.pid,config=str(config),world_size=len(lane['gpus']),gpus=lane['gpus'],
                          started_at=timestamp(),attempt=attempt,command=command,log=str(log_path))
                write_json(record,info)
                write_json(state_path,dict(lane=lane['name'],run=run['id'],status='running',**info))
                code=process.wait()
            step=checkpoint_step(ckpt)
            if code==0 and step==cfg['max_steps']:
                write_json(record,{**info,'status':'completed','finished_at':timestamp(),'step':step})
                write_json(state_path,dict(lane=lane['name'],run=run['id'],status='completed',step=step,at=timestamp()))
                break
            numerical=numerical_failure(out,log_path,log_offset)
            if numerical or attempt==3 or step is None:
                reason=f'exit {code}; latest checkpoint {step}; successor held'
                if numerical: reason+='; numerical failure, automatic retry disabled'
                write_json(out.parent/(out.name+'_stopped.json'),dict(reason=reason,at=timestamp()))
                write_json(state_path,dict(lane=lane['name'],run=run['id'],status='failed',reason=reason,at=timestamp()))
                return False
            write_json(state_path,dict(lane=lane['name'],run=run['id'],status='retrying',step=step,exit_code=code,at=timestamp()))
            time.sleep(30)
    write_json(state_path,dict(lane=lane['name'],status='completed',at=timestamp()))
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True);args=parser.parse_args()
    path=Path(args.plan).resolve();root=path.parent;plan=json.loads(path.read_text())
    with (root/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(root/'supervisor.json',dict(pid=os.getpid(),started_at=timestamp(),plan=str(path),status='running'))
        with ThreadPoolExecutor(max_workers=len(plan['lanes'])) as pool:
            results=list(pool.map(lambda lane:run_lane(lane,plan,root),plan['lanes']))
        write_json(root/'supervisor.json',dict(pid=os.getpid(),finished_at=timestamp(),status='completed' if all(results) else 'failed'))
        if not all(results):raise SystemExit(1)

if __name__=='__main__':main()
