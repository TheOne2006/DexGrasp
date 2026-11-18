"""Disposable four-rank training benchmark with synthetic, pinned CPU batches.

Example: CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/python.sh -m torch.distributed.run
--standalone --nproc-per-node=4 scripts/benchmark_fake.py --model configs/paired_1125/C2.json
--batch 48 --output ../outputs/benchmark/C2-48.json
"""
import argparse
from dataclasses import asdict
from itertools import islice, repeat
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__' and 'LOCAL_RANK' in os.environ:
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', ','.join(map(str, range(int(os.environ['LOCAL_WORLD_SIZE'])))))
    os.environ['CUDA_VISIBLE_DEVICES'] = devices.split(',')[int(os.environ['LOCAL_RANK'])]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from device_memory import DeviceMemory
from dexgrasp.config import DEFAULT_RUN
from dexgrasp.models.config import ModelConfig
from dexgrasp.models.registry import build_model, config_from_settings
from dexgrasp.train.distributed import FMLossModule, streaming_loader
from dexgrasp.train.execution import configure_execution
from dexgrasp.train.prefetch import device_batches
from dexgrasp.train.runner import initialize_checkpoint, source_state
from dexgrasp.data.dataset import sha256


def fake_batch(cfg, batch, seed):
    rng = torch.Generator().manual_seed(seed)
    def points(n):
        xyz = torch.randn(batch, n, 3, generator=rng)
        return nn.functional.normalize(xyz, dim=-1) * 0.4
    condition = {'object_points': points(1024)}
    if cfg.use_scene:
        condition.update(scene_points=points(2048), scene_mask=torch.arange(2048).lt(512).float().expand(batch, -1).contiguous())
    if cfg.use_gap:
        condition[cfg.depth_key] = torch.full((batch, 1), .03)
    if 'grasp_direction_cmd' in cfg.condition_keys:
        condition['grasp_direction_cmd'] = points(1).squeeze(1) / .4
    poses = torch.randn(batch, 3, 21, 3, generator=rng)
    if cfg.stages == 1 and cfg.target != 'grasp':
        poses = poses[:, 1].contiguous()
    return dict(condition={k: v.pin_memory() for k, v in condition.items()}, keypoints_gt=poses.pin_memory())


def run(args):
    local = torch.device('cuda', 0)
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    dist.init_process_group('nccl', device_id=local)
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        torch.manual_seed(42)
        reference = args.model == 'KPGrasp'
        if reference:
            from kpgrasp_benchmark import build_reference, reference_batch
            model, optimizer, architecture = build_reference(args.kpgrasp_root, local, rank, world)
            ddp = DDP(model, device_ids=[0], find_unused_parameters=True)
            obj = fake_batch(ModelConfig(variant='object_only_h_less'), args.batch, 42+rank)['condition']['object_points']
            batch = reference_batch(args.batch, 42+rank, obj)
        else:
            settings = json.loads(Path(args.model).read_text())
            settings = settings.get('model', settings)
            frozen = settings.get('family') == 'frozen_grasp_residual'
            if frozen:
                if not args.frozen_grasp or not args.hand_spec:
                    raise ValueError('benchmarking the frozen model requires --frozen-grasp and --hand-spec')
                spec = json.loads(Path(args.hand_spec).read_text())
            else:
                spec = {'reference_length_m': .204}
            mc = config_from_settings(settings, spec)
            model = nn.SyncBatchNorm.convert_sync_batchnorm(build_model(mc, 'fm')).to(local)
            if frozen:
                initialize_checkpoint(model, args.frozen_grasp, None, sha256(args.hand_spec))
            cfg = dict(DEFAULT_RUN, sparse_convolution_mode=args.sparse_mode, fps_backend='torch' if args.reference_execution else 'triton',
                       select_tokens_before_slice=not args.reference_execution, compile_transformer=args.compile,
                       checkpoint_direction_bias=args.checkpoint_bias)
            configure_execution(model, cfg)
            ddp = DDP(FMLossModule(model), device_ids=[0], broadcast_buffers=False, gradient_as_bucket_view=True)
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.0002, fused=True)
            if frozen and not args.synthetic_hdf5:
                batch = fake_batch(mc.parent, args.batch, 42+rank)
                if mc.parent.use_gap:
                    batch['condition']['h32'] = batch['condition'].pop('palm_gap')
                batch['keypoints_gt'] = torch.randn(args.batch, 3, 21, 3,
                    generator=torch.Generator().manual_seed(42+rank)).pin_memory()
            elif not args.synthetic_hdf5:
                batch = fake_batch(mc, args.batch, 42+rank)
            architecture = asdict(mc)
        loader = None
        input_description = 'pinned synthetic sphere repeated; fresh FM noise/time'
        if args.synthetic_hdf5:
            if reference or mc.stages != 3:
                raise ValueError('the paired synthetic HDF5 option requires a DataV2 model')
            cfg.update(dataset_root=args.synthetic_hdf5, dataset_format='data_v2', batch_size=args.batch,
                       accumulation=1, num_workers=args.workers)
            if args.replay_hdf5_batch:
                # Keep the same point distribution and H2D path as the HDF5 run.
                cfg['num_workers'] = 0
                fixture = streaming_loader(cfg, model, world, rank, 0, 1)
                try:
                    batch = next(iter(fixture))
                finally:
                    fixture.dataset.close()
                input_description = 'first paired HDF5 batch repeated from pinned CPU memory'
            else:
                loader = streaming_loader(cfg, model, world, rank, 0, args.warmup+args.steps)
                host_stream = iter(loader)
                input_description = 'paired HDF5 with production sampler, workers and pinned batches'
        torch.manual_seed(42 + rank)

        timings = []
        def updates(n, *, measure=False):
            loss_sum = torch.zeros((), device=local)
            batches = repeat(batch, n) if loader is None else islice(host_stream, n)
            for data in device_batches(batches, local, prefetch=not args.reference_execution):
                if measure:
                    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                    events[0].record()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=not reference):
                    result = ddp(data)
                    loss = result['loss_fm'] if reference else result
                if measure:
                    events[1].record()
                loss.backward()
                if measure:
                    events[2].record()
                nn.utils.clip_grad_norm_(model.parameters(), 10. if reference else 1., error_if_nonfinite=True)
                optimizer.step()
                loss_sum += loss.detach()
                if measure:
                    events[3].record()
                    timings.append(events)
            return loss_sum

        gpu_uuid = 'GPU-' + str(torch.cuda.get_device_properties(0).uuid)
        monitor = DeviceMemory(gpu_uuid)
        updates(args.warmup)
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        loss_sum = updates(args.steps, measure=args.phase_timings)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        peak = dict(rank=rank, gpu=os.environ['CUDA_VISIBLE_DEVICES'], seconds=seconds,
                    allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                    reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
                    total_GiB=torch.cuda.get_device_properties(0).total_memory/2**30,
                    loss=float(loss_sum)/args.steps)
        monitor.close()
        peak.update(device_used_peak_MiB=monitor.peak/2**20, device_total_MiB=monitor.total/2**20)
        if timings:
            peak['mean_phase_ms'] = {
                name: sum(row[i].elapsed_time(row[i+1]) for row in timings)/len(timings)
                for i, name in enumerate(('forward', 'backward_ddp', 'clip_optimizer'))
            }
        monitor = None
        all_peaks = [None]*world
        dist.all_gather_object(all_peaks, peak)
        if rank == 0:
            report = dict(model=args.model, settings=architecture, batch_per_gpu=args.batch, global_batch=world*args.batch,
                          samples_per_second=world*args.batch*args.steps/max(p['seconds'] for p in all_peaks),
                          ranks=all_peaks, torch_version=str(torch.__version__), cuda=torch.version.cuda,
                          allocator=torch.cuda.memory.get_allocator_backend(),
                          allocator_config=os.environ.get('PYTORCH_ALLOC_CONF', os.environ.get('PYTORCH_CUDA_ALLOC_CONF')),
                          source=source_state(), host=platform.node(), nccl=torch.cuda.nccl.version(),
                          transport={k:os.environ.get(k) for k in ('NCCL_P2P_DISABLE', 'NCCL_CUMEM_ENABLE',
                                                                    'NCCL_CUMEM_HOST_ENABLE', 'NCCL_IB_DISABLE')},
                          settings_benchmark=vars(args), seed=42, precision='FP32' if reference else 'BF16 dense / FP32 sparse',
                          input=input_description)
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x') as f:
                json.dump(report, f, indent=2, allow_nan=False)
            print(json.dumps({k:report[k] for k in ('model','global_batch','samples_per_second')}),flush=True)
    finally:
        if 'monitor' in locals() and monitor is not None:
            monitor.close()
        if 'loader' in locals() and loader is not None:
            loader.dataset.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='Model-only or historical training JSON')
    p.add_argument('--synthetic-hdf5', help='Time the production stream from an existing synthetic DataV2 fixture')
    p.add_argument('--replay-hdf5-batch', action='store_true', help='Repeat the first pinned HDF5 batch to isolate loading overhead')
    p.add_argument('--phase-timings', action='store_true', help='Record CUDA events for each measured update phase')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--sparse-mode', choices=('default','direct','copy','copy_hidden'), default='default')
    p.add_argument('--frozen-grasp')
    p.add_argument('--hand-spec')
    p.add_argument('--kpgrasp-root', default=str(ROOT.parent/'reference/KPGrasp'))
    p.add_argument('--batch', required=True, type=int)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--output', required=True)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--checkpoint-bias', action='store_true')
    p.add_argument('--reference-execution', action='store_true')
    args = p.parse_args()
    if min(args.batch, args.warmup, args.steps) < 1:
        p.error('batch, warmup and steps must be positive')
    if args.replay_hdf5_batch and not args.synthetic_hdf5:
        p.error('--replay-hdf5-batch requires --synthetic-hdf5')
    run(args)
