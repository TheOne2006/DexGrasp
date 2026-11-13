"""Measure the streaming input pipeline without GPU compute or optimizer updates.

First-batch latency includes process startup. OS page cache is never flushed.
Launch with torchrun for realistic concurrent rank/worker storage contention.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from dexgrasp.config import load_config, model_config
from dexgrasp.geometry.hand import URDFHand
from dexgrasp.train.distributed import streaming_loader
from dexgrasp.train.runner import source_state


def process_stats(pids):
    total=dict(read_bytes=0,rchar=0,rss_kib=0,major_faults=0,minor_faults=0)
    for pid in pids:
        root=Path('/proc')/str(pid)
        try:
            io=dict(line.split(':',1) for line in (root/'io').read_text().splitlines())
            total['read_bytes']+=int(io['read_bytes']);total['rchar']+=int(io['rchar'])
            for line in (root/'status').read_text().splitlines():
                if line.startswith('VmRSS:'):total['rss_kib']+=int(line.split()[1])
            stat=(root/'stat').read_text().rsplit(')',1)[1].split()
            total['minor_faults']+=int(stat[7]);total['major_faults']+=int(stat[9])
        except FileNotFoundError:pass
    return total


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=30)
    args=p.parse_args();cfg=load_config(args.config)
    if args.steps<2 or cfg['accumulation']!=1:raise ValueError('steps>=2, accumulation=1 required')
    rank,world=int(os.environ.get('RANK',0)),int(os.environ.get('WORLD_SIZE',1))
    torch.set_num_threads(cfg['cpu_threads'])
    if torch.cuda.is_available():torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
    if world>1:dist.init_process_group('gloo')
    try:
        mc=model_config(cfg,URDFHand(cfg['hand_spec']).spec)
        start=time.perf_counter();loader=streaming_loader(cfg,SimpleNamespace(cfg=mc),world,rank,0,args.steps)
        iterator=iter(loader)
        pids=[os.getpid()]+[w.pid for w in (getattr(iterator,'_workers',None) or [])]
        before=process_stats(pids);latencies=[]
        first_ready=None
        for i in range(args.steps):
            tick=time.perf_counter();batch=next(iterator)
            latencies.append(time.perf_counter()-tick)
            if i==0:first_ready=time.perf_counter()-start
            if len(batch['keypoints_gt'])!=cfg['batch_size']:raise AssertionError('short batch')
        elapsed=time.perf_counter()-start;after=process_stats(pids)
        report=dict(rank=rank,workers=cfg['num_workers'],first_batch_seconds=first_ready,wall_seconds=elapsed,
                    batch_wait_seconds=latencies,steady_samples_per_second=(args.steps-1)*cfg['batch_size']/sum(latencies[1:]),
                    final_rss_GiB=after['rss_kib']/2**20,counters_delta={k:after[k]-before[k] for k in before if k!='rss_kib'})
        reports=[report]
        if world>1:
            reports=[None]*world;dist.all_gather_object(reports,report)
        if rank==0:
            result=dict(config=cfg,source=source_state(),world_size=world,ranks=reports,
                        aggregate_samples_per_second=world*cfg['batch_size']*args.steps/max(r['wall_seconds'] for r in reports),
                        cache_policy='No OS cache flush. First process pass and subsequent passes must be labeled separately.')
            with Path(args.output).open('x') as f:json.dump(result,f,indent=2)
            print(json.dumps({k:result[k] for k in ('world_size','aggregate_samples_per_second')}),flush=True)
    finally:
        if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
