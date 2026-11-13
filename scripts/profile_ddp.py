"""Bounded DDP profiling: separate uninstrumented throughput and diagnostic CUDA trace."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from dexgrasp.config import load_config, model_config
from dexgrasp.geometry.hand import URDFHand
from dexgrasp.models.keypoint import KeypointModel
from dexgrasp.train.distributed import FMLossModule, streaming_loader, state_digest
from dexgrasp.train.execution import configure_execution
from dexgrasp.train.optimizers import make_optimizer
from dexgrasp.train.runner import move_batch, source_state, runtime_identity
from benchmark_input import process_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--profile-steps', type=int, default=2)
    args = parser.parse_args()
    if min(args.warmup, args.steps, args.profile_steps) < 1:
        raise ValueError('positive iteration counts required')
    cfg = load_config(args.config)
    if cfg['accumulation'] != 1 or cfg['data_loading'] != 'stream':
        raise ValueError('profile uses streaming with accumulation=1')
    rank, world, local = (int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.set_num_threads(cfg['cpu_threads'])
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(cfg['seed'])
    dist.init_process_group('nccl', device_id=device)
    out = Path(args.output_dir)
    try:
        if rank == 0:
            out.mkdir(parents=True, exist_ok=False)
            (out/'provenance.json').write_text(json.dumps(dict(config=cfg,source=source_state(),runtime=runtime_identity(),scope='disposable real optimizer updates; no quality evaluation'),indent=2))
        dist.barrier()
        hand = URDFHand(cfg['hand_spec'])
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(KeypointModel(model_config(cfg, hand.spec)).to(device))
        ddp = DistributedDataParallel(FMLossModule(model), device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True)
        configure_execution(model, cfg)
        optimizer = make_optimizer(model, cfg, fused=True)
        count = args.warmup+args.steps+args.profile_steps
        loader = streaming_loader(cfg, model, world, rank, 0, count)
        batches = iter(loader)
        torch.manual_seed(cfg['seed']+rank*100003)
        records = []
        def iteration(record=False, instrument=False):
            mark = torch.profiler.record_function if instrument else lambda _: nullcontext()
            begin = time.perf_counter()
            with mark('phase/input_wait'):
                cpu_batch = next(batches)
            host_wait = time.perf_counter()-begin
            events = [torch.cuda.Event(enable_timing=True) for _ in range(5)] if record else None
            if record: events[0].record()
            with mark('phase/H2D'):
                batch = move_batch(cpu_batch, device)
            if record: events[1].record()
            optimizer.zero_grad(set_to_none=True)
            with mark('phase/forward'), torch.autocast('cuda', dtype=torch.bfloat16):
                loss = ddp(batch)
            if record: events[2].record()
            with mark('phase/backward_DDP'):
                loss.backward()
            if record: events[3].record()
            with mark('phase/clip_optimizer'):
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
                optimizer.step()
            if record:
                events[4].record(); events[4].synchronize()
                records.append(dict(input_wait_ms=1000*host_wait, h2d_ms=events[0].elapsed_time(events[1]), forward_ms=events[1].elapsed_time(events[2]), backward_ddp_ms=events[2].elapsed_time(events[3]), optimizer_ms=events[3].elapsed_time(events[4]), wall_ms=1000*(time.perf_counter()-begin),loss=float(loss.detach()),grad_norm=float(norm)))
        for _ in range(args.warmup): iteration()
        torch.cuda.synchronize();dist.barrier();torch.cuda.reset_peak_memory_stats()
        pids=[os.getpid()]+[w.pid for w in (getattr(batches,'_workers',None) or [])]
        io_before=process_stats(pids)
        start=time.perf_counter()
        for _ in range(args.steps): iteration(record=True)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        memory=dict(allocated_GiB=torch.cuda.max_memory_allocated()/2**30,reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
        io_after=process_stats(pids)
        result=dict(rank=rank,elapsed=elapsed,records=records,memory=memory,
                    input_process_counters={k:io_after[k]-io_before[k] for k in io_before if k!='rss_kib'},
                    input_process_sum_RSS_GiB=io_after['rss_kib']/2**20)
        all_ranks=[None]*world;dist.all_gather_object(all_ranks,result)
        if rank==0:
            summary=dict(world_size=world,local_batch=cfg['batch_size'],steps=args.steps,samples_per_second=world*cfg['batch_size']*args.steps/max(x['elapsed'] for x in all_ranks),ranks=all_ranks)
            (out/'timings.json').write_text(json.dumps(summary,indent=2))
            print(json.dumps({k:summary[k] for k in ('samples_per_second','steps')}),flush=True)
        # Module hooks exist only during the diagnostic trace, outside throughput timing.
        handles=[]; contexts={}
        for name,module in [('backbone',model.object_encoder.backbone),('encoder',model.object_encoder)]:
            def before(module, inputs, label=name):
                context=torch.profiler.record_function('module/'+label);context.__enter__();contexts[label]=context
            def after(module, inputs, output, label=name):contexts.pop(label).__exit__(None,None,None)
            handles.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
        dist.barrier()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True,with_stack=False) as profiler:
            for _ in range(args.profile_steps):iteration(instrument=True);profiler.step()
        for handle in handles:handle.remove()
        profiler.export_chrome_trace(str(out/f'rank{rank}.trace.json.gz'))
        (out/f'rank{rank}.operators.txt').write_text(profiler.key_averages().table(sort_by='self_cuda_time_total',row_limit=70))
        operators=[]
        for event in profiler.key_averages():
            operators.append(dict(name=event.key,calls=event.count,self_cpu_us=event.self_cpu_time_total,self_device_us=event.self_device_time_total,total_device_us=event.device_time_total))
        (out/f'rank{rank}.operators.json').write_text(json.dumps(operators,indent=2))
        missing=[name for name,param in model.named_parameters() if param.grad is None]
        checks=[None]*world
        dist.all_gather_object(checks,dict(rank=rank,model_digest=state_digest(model),missing_gradients=missing))
        if any(x['missing_gradients'] for x in checks) or len({x['model_digest'] for x in checks})!=1:
            raise AssertionError('Disconnected parameters or inconsistent rank states')
        if rank==0:(out/'rank_consistency.json').write_text(json.dumps(checks,indent=2))
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__=='__main__':main()
