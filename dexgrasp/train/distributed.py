"""Single-node FM DDP: streamed full batches, optional small-data GPU cache, exact resume."""
import argparse
import ctypes
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from ..config import load_config, model_config
from ..data.dataset import GraspDataset, audit_dataset, sha256
from ..data.audit_receipt import verified_identity
from ..geometry.hand import URDFHand
from ..models.encoders import farthest_point_indices
from ..models.keypoint import KeypointModel
from .objectives import fm_loss
from .optimizers import make_optimizer
from .schedule import apply_learning_rate
from .execution import configure_execution
from .runner import atomic_checkpoint, initialize_checkpoint, runtime_identity, source_state, move_batch
from .checkpoints import ARCHIVE_EVERY
from .recovery import validate_recovery


class GlobalBatchStream:
    """Shard positions in a common infinite stream of shuffled training epochs.

    A global batch larger than the dataset necessarily repeats examples. Rank
    shards never share stream positions; repeats come only from cycling epochs.
    """
    def __init__(self, size, local_batch, world_size, rank, seed):
        if min(size, local_batch, world_size) < 1 or not 0 <= rank < world_size:
            raise ValueError('invalid global batch stream')
        self.size, self.local_batch, self.world_size = size, local_batch, world_size
        self.rank, self.seed, self.orders = rank, seed, {}

    def indices(self, step):
        if step < 0:
            raise ValueError('negative sampler step')
        start = step*self.local_batch*self.world_size + self.rank*self.local_batch
        end = start+self.local_batch
        values = []
        while start < end:
            epoch, offset = divmod(start, self.size)
            if epoch not in self.orders:
                self.orders[epoch] = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed+epoch))
            count = min(end-start, self.size-offset)
            values.extend(self.orders[epoch][offset:offset+count].tolist())
            start += count
        self.orders = {k: v for k, v in self.orders.items() if k >= (end-self.local_batch)//self.size}
        return values


class StreamBatchSampler:
    """Resume from consumed microbatches, independent of worker prefetch."""
    def __init__(self, stream, start, stop):
        self.stream, self.start, self.stop = stream, start, stop

    def __iter__(self):
        for cursor in range(self.start, self.stop):
            yield self.stream.indices(cursor)

    def __len__(self):
        return self.stop - self.start


def training_collate(samples):
    if isinstance(samples, dict):
        return samples  # Already batched tensors from packed table gathers.
    # Metadata strings are useful for evaluation, but waste worker IPC in training.
    return default_collate([{k: sample[k] for k in ('condition', 'keypoints_gt')} for sample in samples])


def streaming_loader(cfg, model, world, rank, start, stop):
    if cfg.get('dataset_format', 'data_v1') == 'data_v1':
        dataset = GraspDataset(cfg['dataset_root'], 'train', model.cfg.coord_scale_m,
                               observation_cache_size=cfg['observation_cache_size'],
                               max_open_shards=cfg['max_open_shards'], condition_keys=model.cfg.condition_keys, training_only=True)
    else:
        from ..data.packed import PackedTrainingDataset
        from ..data.object_corpus import ObjectCorpusDataset
        factory = PackedTrainingDataset if cfg['dataset_format'] == 'packed' else ObjectCorpusDataset
        dataset = factory(cfg['dataset_root'], 'train', model.cfg.coord_scale_m, condition_keys=model.cfg.condition_keys)
    stream = GlobalBatchStream(len(dataset), cfg['batch_size'], world, rank, cfg['seed'])
    sampler = StreamBatchSampler(stream, start*cfg['accumulation'], stop*cfg['accumulation'])
    options = dict(num_workers=cfg['num_workers'], pin_memory=True, collate_fn=training_collate,
                   generator=torch.Generator().manual_seed(cfg['seed']+rank))
    if cfg['num_workers']:
        options.update(prefetch_factor=cfg['prefetch_factor'], persistent_workers=True,
                       multiprocessing_context='spawn')
    return DataLoader(dataset, batch_sampler=sampler, **options)


def local_device_index():
    # scripts/train_fm_ddp.py isolates one GPU before Torch import. Also support
    # direct module launches where all local GPUs remain visible.
    return 0 if os.environ.get('DEXGRASP_ISOLATED_GPU') == '1' else int(os.environ.get('LOCAL_RANK', 0))


class FMLossModule(nn.Module):
    """Encoding and the velocity network must both execute inside DDP.forward."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return fm_loss(self.model, batch)


def cache_training_data(cfg, model):
    # This optimization is valid for this fixed, precomputed dataset: no online augmentation.
    device = next(model.parameters()).device
    dataset = GraspDataset(cfg['dataset_root'], 'train', model.cfg.coord_scale_m,
                           observation_cache_size=cfg['observation_cache_size'],
                           max_open_shards=cfg['max_open_shards'], condition_keys=model.cfg.condition_keys, training_only=True)
    chunks, targets = {}, []
    try:
        for batch in DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0):
            for key, tensor in batch['condition'].items():
                chunks.setdefault(key, []).append(tensor)
            targets.append(batch['keypoints_gt'])
    finally:
        dataset.close()
    condition = {key: torch.cat(values).to(device) for key, values in chunks.items()}
    for prefix in (('object', 'scene') if model.cfg.use_scene else ('object',)):
        points = condition[prefix+'_points']
        condition[prefix+'_fps_indices'] = torch.cat([
            farthest_point_indices(chunk, model.cfg.tokens) for chunk in points.split(128)])
    return dict(condition=condition, keypoints_gt=torch.cat(targets).to(device))


def cached_batch(cache, indices):
    index = torch.tensor(indices, dtype=torch.long, device=cache['keypoints_gt'].device)
    return dict(condition={k: v.index_select(0, index) for k, v in cache['condition'].items()},
                keypoints_gt=cache['keypoints_gt'].index_select(0, index))


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state(state['cuda'])


def state_digest(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(cfg, *, resume=None, stop_after=None, fork_from=None):
    if resume and fork_from:
        raise ValueError('resume and fork_from are mutually exclusive')
    if cfg.get('dataset_format', 'data_v1') == 'object_corpus':
        raise ValueError('Formal corpus training requires audited packed tables')
    if cfg.get('dataset_format') == 'packed' and cfg['data_loading'] != 'stream':
        raise ValueError('Packed corpus requires streaming input')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world < 1 or not 0 <= rank < world:
        raise ValueError('invalid distributed world/rank')
    global_batch = cfg['batch_size'] * world * cfg['accumulation']
    if cfg['method'] != 'fm' or cfg['freeze_encoders'] or cfg['overfit_diagnostic']:
        raise ValueError('This runner trains all FM parameters and defers reconstruction evaluation')
    if cfg['precision'] != 'bfloat16' or not torch.cuda.is_bf16_supported():
        raise ValueError('This DDP experiment requires supported BF16')
    torch.set_num_threads(cfg['cpu_threads'])
    device_index = local_device_index()
    torch.cuda.set_device(device_index)
    device = torch.device('cuda', device_index)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    dist.init_process_group('nccl', timeout=timedelta(minutes=10), device_id=device)
    print(json.dumps(dict(rank=rank, physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                          phase='nccl_ready', pid=os.getpid())), flush=True)
    try:
        hand = URDFHand(cfg['hand_spec']); hand_hash = sha256(cfg['hand_spec'])
        if hand_hash != sha256(Path(cfg['dataset_root'])/'hand/hand_spec.json'):
            raise ValueError('hand spec mismatch')
        identity_box = [None]
        if rank == 0:
            try:
                scale = cfg['model'].get('coord_scale_m', 0.2)
                if cfg.get('dataset_format') == 'packed':
                    from ..data.formal_corpus import verified_packed_identity
                    identity_box[0] = verified_packed_identity(cfg['dataset_root'], scale)
                else:
                    identity_box[0] = (verified_identity(cfg['dataset_root'], scale, cfg['data_audit'])
                                   if cfg.get('data_audit') else audit_dataset(cfg['dataset_root'], scale, deep=True))
                identity_box[0]['hand_urdf_sha256'] = sha256(hand.urdf_path)
            except Exception as exc:
                identity_box[0] = {'audit_error': f'{type(exc).__name__}: {exc}'}
        dist.broadcast_object_list(identity_box, src=0)
        identity = identity_box[0]
        if 'audit_error' in identity:
            raise ValueError(identity['audit_error'])
        source, runtime = source_state(), runtime_identity()
        nccl_libraries = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                          if '/libnccl.so' in line}
        # torch.cuda.nccl.version reports the version used to compile Torch.
        nccl_version = ctypes.c_int()
        if ctypes.CDLL(sorted(nccl_libraries)[0]).ncclGetVersion(ctypes.byref(nccl_version)) != 0:
            raise RuntimeError('Cannot query loaded NCCL version')
        runtime['nccl_runtime_version'] = nccl_version.value
        runtime['nccl_torch_build_version'] = tuple(torch.cuda.nccl.version())
        runtime['nccl_libraries'] = {path: sha256(path) for path in sorted(nccl_libraries)}
        runtime['nccl_environment'] = {key: os.environ.get(key) for key in
                                       ('NCCL_P2P_DISABLE', 'NCCL_CUMEM_HOST_ENABLE', 'NCCL_CUMEM_ENABLE')}
        mc = model_config(cfg, hand.spec)
        random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
        model = KeypointModel(mc, 'fm').to(device)
        if cfg['initialization'] not in ('random', 'checkpoint'):
            raise ValueError('DDP supports random or same-identity checkpoint initialization')
        initial_hash = (initialize_checkpoint(model, cfg['initial_checkpoint'], identity, hand_hash)
                        if cfg['initialization'] == 'checkpoint' else None)
        # ME's batch norm wraps torch BatchNorm1d; recursive conversion synchronizes sparse feature statistics.
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        optimizer = make_optimizer(model, cfg, fused=True)
        start = 0
        resumed = None
        recovery = None
        if resume or fork_from:
            resumed = torch.load(resume or fork_from, map_location='cpu', weights_only=False)
            if fork_from:
                validate_recovery(resumed, cfg, identity, runtime, world, hand_hash, initial_hash)
                recovery = dict(parent_checkpoint=str(Path(fork_from).resolve()),
                                parent_sha256=sha256(fork_from), parent_step=resumed['step'],
                                parent_source=resumed['source'], parent_config=resumed['config'],
                                preserved=['model', 'optimizer', 'rank_rng', 'sampler_cursor', 'lr_schedule'])
            elif (resumed['config'] != cfg or resumed['data_identity'] != identity
                    or resumed['source']['source_sha256'] != source['source_sha256']
                    or resumed['runtime'] != runtime or resumed['initial_checkpoint_sha256'] != initial_hash
                    or resumed['distributed']['world_size'] != world):
                raise ValueError('DDP resume requires identical config/data/source/runtime/parent/world size')
            else:
                recovery = resumed.get('recovery')
            model.load_state_dict(resumed['model']); optimizer.load_state_dict(resumed['optimizer'])
            start = resumed['step']
        if not start < cfg['max_steps'] or (stop_after is not None and not start < stop_after <= cfg['max_steps']):
            raise ValueError('invalid starting or stopping step')
        run_dir, ckpt_dir = Path(cfg['run_dir']), Path(cfg['checkpoint_dir'])
        if not resume and any(p.exists() and any(p.iterdir()) for p in (run_dir, ckpt_dir)):
            raise FileExistsError('Use new run/checkpoint directories')
        ddp = DistributedDataParallel(FMLossModule(model), device_ids=[device_index], output_device=device_index,
                                      broadcast_buffers=False, gradient_as_bucket_view=True,
                                      bucket_cap_mb=25, find_unused_parameters=False)
        configure_execution(model, cfg)
        model.train()
        cache, loader, batches = None, None, None
        if cfg['data_loading'] == 'gpu_cache':
            cache = cache_training_data(cfg, model)
            sampler = GlobalBatchStream(len(cache['keypoints_gt']), cfg['batch_size'], world, rank, cfg['seed'])
        else:
            loader = streaming_loader(cfg, model, world, rank, start, stop_after or cfg['max_steps'])
            batches = iter(loader)
        print(json.dumps(dict(rank=rank, phase='input_ready', data_loading=cfg['data_loading'])), flush=True)
        random.seed(cfg['seed']+rank*100003); np.random.seed(cfg['seed']+rank*100003); torch.manual_seed(cfg['seed']+rank*100003)
        if resumed is not None:
            restore_rng(resumed['distributed']['rank_rng'][rank])
            del resumed
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True); ckpt_dir.mkdir(parents=True, exist_ok=True)
            if not resume:
                provenance = dict(config=cfg, model_config=asdict(mc), source=source, runtime=runtime,
                                  data_identity=identity, hand_spec=hand.spec, initial_checkpoint_sha256=initial_hash,
                                  initialization=cfg['initialization'], trainable_parameters=sum(p.numel() for p in model.parameters()),
                                  recovery=recovery,
                                  frozen_parameters=0, physical_success_evaluated=False,
                                  distributed=dict(world_size=world, backend='nccl', local_batch_size=cfg['batch_size'],
                                                   global_batch_size=global_batch, sync_batchnorm=True,
                                                   accumulation=cfg['accumulation'], gpu_input_cache=cache is not None, cached_fps=cache is not None, gradient_as_bucket_view=True,
                                                   optimizer=cfg['optimizer'], fused_adamw=True, tf32=torch.backends.cuda.matmul.allow_tf32, precision=cfg['precision'],
                                                   sampler='global shuffled epoch stream; repeats only across epoch boundaries'),
                                  checkpoint_selection_metric='last_optimizer_step; generation evaluation deferred')
                (run_dir/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
        dist.barrier()
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        start_clock = time.monotonic(); segment_clock = start_clock; log_every = 10
        losses = torch.zeros((), device=device); segment_count = 0
        window_norm_max = torch.zeros((), device=device)
        log = (run_dir/'metrics.jsonl').open('a') if rank == 0 else None
        try:
            for step in range(start+1, cfg['max_steps']+1):
                learning_rates = apply_learning_rate(optimizer, cfg, step-1)
                optimizer.zero_grad(set_to_none=True)
                step_loss = torch.zeros((), device=device)
                for micro in range(cfg['accumulation']):
                    cursor = (step-1)*cfg['accumulation']+micro
                    batch = (cached_batch(cache, sampler.indices(cursor)) if cache is not None
                             else move_batch(next(batches), device))
                    sync = nullcontext() if micro == cfg['accumulation']-1 else ddp.no_sync()
                    with sync:
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            loss = ddp(batch) / cfg['accumulation']
                        loss.backward()
                    step_loss += loss.detach().float()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
                if cfg.get('abort_grad_norm', 0) and norm.item() > cfg['abort_grad_norm']:
                    if rank == 0:
                        (run_dir/'numerical_failure.json').write_text(json.dumps(dict(
                            step=step, grad_norm=float(norm), loss=float(step_loss),
                            threshold=cfg['abort_grad_norm'], optimizer_step_applied=False), indent=2)+'\n')
                    raise FloatingPointError(f'gradient norm guard at step {step}: {float(norm)}')
                optimizer.step()
                window_norm_max = torch.maximum(window_norm_max, norm.detach())
                losses += step_loss; segment_count += 1
                checkpoint_due = step % cfg['save_every'] == 0 or step == cfg['max_steps'] or step == stop_after
                if step % log_every == 0 or checkpoint_due:
                    dist.all_reduce(losses, op=dist.ReduceOp.SUM)
                    torch.cuda.synchronize(); now = time.monotonic()
                    mean_loss = float(losses)/(world*segment_count)
                    if not np.isfinite(mean_loss):
                        raise FloatingPointError('nonfinite distributed training loss')
                    if log is not None:
                        record = dict(step=step, loss=mean_loss, loss_window_steps=segment_count, lr=learning_rates['adamw'], learning_rates=learning_rates,
                                      grad_norm=float(norm), grad_norm_window_max=float(window_norm_max), seconds=now-start_clock,
                                      seconds_per_step=(now-segment_clock)/segment_count,
                                      global_batch_size=global_batch, world_size=world,
                                      samples_per_second=global_batch*segment_count/(now-segment_clock))
                        log.write(json.dumps(record, allow_nan=False)+'\n'); log.flush()
                    losses.zero_(); segment_count = 0; segment_clock = time.monotonic()
                    window_norm_max.zero_()
                if checkpoint_due:
                    states = [None]*world
                    dist.all_gather_object(states, rng_state())
                    if step == stop_after:
                        checks = [None]*world
                        dist.all_gather_object(checks, dict(rank=rank, digest=state_digest(model),
                            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                            peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30))
                        if len({c['digest'] for c in checks}) != 1:
                            raise RuntimeError('DDP ranks disagree on trained model or BN buffers')
                        if rank == 0:
                            (run_dir/'rank_consistency.json').write_text(json.dumps(checks, indent=2)+'\n')
                    if rank == 0:
                        state = dict(schema='dexgrasp-v1', method='fm', step=step, config=cfg,
                                     model_config=asdict(mc), model=model.state_dict(), optimizer=optimizer.state_dict(),
                                     scheduler=dict(kind=cfg['lr_schedule'], completed_steps=step, learning_rates=learning_rates),
                                     source=source, data_identity=identity, hand_spec_sha256=hand_hash,
                                     teacher_sha256=None, best_val=None, validation_metric='deferred',
                                     checkpoint_selection_metric='last_optimizer_step',
                                     initial_checkpoint_sha256=initial_hash, runtime=runtime,
                                     recovery=recovery,
                                     torch_rng=states[0]['torch'], cuda_rng=[states[0]['cuda']],
                                     distributed=dict(world_size=world, rank_rng=states, sync_batchnorm=True,
                                                      sampler='global-epoch-stream-v1', global_batch_size=global_batch))
                        atomic_checkpoint(ckpt_dir/'last.pt', state)
                    dist.barrier()
                    segment_clock = time.monotonic()
                if step == stop_after:
                    break
        finally:
            if log is not None: log.close()
            if loader is not None:
                loader.dataset.close()
        return dict(status='completed' if step == cfg['max_steps'] else 'paused', step=step,
                    world_size=world, local_batch_size=cfg['batch_size'], global_batch_size=global_batch)
    finally:
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--resume'); p.add_argument('--stop-after', type=int)
    p.add_argument('--fork-from', help='explicit numerical-repair fork retaining optimizer/RNG/cursor')
    args = p.parse_args()
    result = run(load_config(args.config), resume=args.resume, stop_after=args.stop_after, fork_from=args.fork_from)
    if int(os.environ['RANK']) == 0:
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
