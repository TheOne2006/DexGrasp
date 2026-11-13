"""Bounded real-backbone DDP throughput check; no optimizer updates by default."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from dexgrasp.config import load_config, model_config
from dexgrasp.geometry.hand import URDFHand
from dexgrasp.train.distributed import FMLossModule, cache_training_data, cached_batch, GlobalBatchStream, streaming_loader
from dexgrasp.train.runner import source_state, runtime_identity, move_batch
from dexgrasp.models.keypoint import KeypointModel
from dexgrasp.train.optimizers import make_optimizer
from dexgrasp.train.execution import configure_execution


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--updates', action='store_true', help='Disposable optimizer update validation; not a saved training run')
    args = p.parse_args()
    if args.warmup < 1 or args.steps < 1:
        raise ValueError('positive warmup/steps required')
    cfg = load_config(args.config)
    if cfg['accumulation'] != 1:
        raise ValueError('benchmark measures one microbatch; use production runner for accumulation validation')
    rank, world, local = (int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    torch.set_num_threads(cfg['cpu_threads'])
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(cfg['seed'])
    dist.init_process_group('nccl', device_id=device)
    try:
        hand = URDFHand(cfg['hand_spec'])
        mc = model_config(cfg, hand.spec)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(KeypointModel(mc).to(device))
        ddp = DDP(FMLossModule(model), device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True)
        configure_execution(model, cfg)
        optimizer = make_optimizer(model, cfg, fused=True)
        if cfg['data_loading'] == 'gpu_cache':
            cache = cache_training_data(cfg, model)
            sampler = GlobalBatchStream(len(cache['keypoints_gt']), cfg['batch_size'], world, rank, cfg['seed'])
            get_batch = lambda i: cached_batch(cache, sampler.indices(i))
        else:
            loader = streaming_loader(cfg, model, world, rank, 0, args.warmup+args.steps)
            batches = iter(loader)
            get_batch = lambda i: move_batch(next(batches), device)
        torch.manual_seed(cfg['seed']+rank*100003)
        timings, waits, losses = [], [], []
        torch.cuda.reset_peak_memory_stats()
        for i in range(args.warmup+args.steps):
            dist.barrier()
            torch.cuda.synchronize()
            start = time.perf_counter()
            batch = get_batch(i)
            torch.cuda.synchronize()
            loaded = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = ddp(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
            if args.updates:
                optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            values = torch.tensor([elapsed, loaded-start], device=device)
            dist.all_reduce(values, op=dist.ReduceOp.MAX)
            if i >= args.warmup:
                timings.append(float(values[0])); waits.append(float(values[1])); losses.append(float(loss.detach()))
        # Check that EVERY parameter participates and all-reduced gradients match.
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        error = torch.zeros((), device=device)
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            reference = parameter.grad.clone()
            dist.broadcast(reference, 0)
            error = torch.maximum(error, (reference-parameter.grad).abs().max())
        dist.all_reduce(error, op=dist.ReduceOp.MAX)
        if missing or float(error) != 0:
            raise AssertionError(f'disconnected parameters={missing}; gradient error={error}')
        memory = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()], device=device, dtype=torch.float64)
        dist.all_reduce(memory, op=dist.ReduceOp.MAX)
        if rank == 0:
            report = dict(scope='bounded software throughput; no grasp quality evaluation', config=cfg,
                          model_config=asdict(mc), world_size=world, samples_per_second=world*cfg['batch_size']*len(timings)/sum(timings),
                          seconds_per_step=timings, input_seconds=waits, losses=losses, warmup=args.warmup,
                          peak_allocated_GiB=float(memory[0])/2**30,
                          peak_reserved_GiB=float(memory[1])/2**30,
                          optimizer_updates=(args.warmup+args.steps if args.updates else 0),
                          missing_gradients=missing, max_rank_gradient_error=float(error),
                          source=source_state(), runtime=runtime_identity())
            output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
            with output.open('x') as f: json.dump(report, f, indent=2)
            print(json.dumps({k: report[k] for k in ('samples_per_second','peak_allocated_GiB','max_rank_gradient_error','optimizer_updates')}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
