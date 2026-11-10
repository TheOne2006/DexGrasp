"""Single-device v1 training, deterministic resume and frozen-teacher online distillation."""
from dataclasses import asdict
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from ..config import ModelConfig, model_config
from ..data.dataset import GraspDataset, audit_dataset, sha256
from ..geometry.hand import URDFHand
from ..models.encoders import GeometryEncoder
from ..models.keypoint import KeypointModel
from .objectives import fm_loss, flowmap_loss, sample_keypoints, evaluation


def source_state():
    root = Path(__file__).resolve().parents[1]
    repo = root.parent
    result = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    status = subprocess.run(['git', '-C', str(repo), 'status', '--short'], capture_output=True, text=True)
    files = sorted(root.rglob('*.py')) + sorted((repo/'scripts/patches').glob('*.patch')) + sorted((repo/'scripts').glob('*.py'))
    return dict(git_commit=result.stdout.strip() if result.returncode == 0 else None,
                git_status=status.stdout.strip(),
                source_sha256={str(p.relative_to(repo)): sha256(p) for p in files})


def runtime_identity():
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'h5py', 'scipy')}
    versions['torch_cuda'] = torch.version.cuda
    versions['minkowski_extension_sha256'] = None
    if importlib.util.find_spec('MinkowskiEngineBackend') is not None:
        spec = importlib.util.find_spec('MinkowskiEngineBackend._C')
        if spec is not None and spec.origin:
            versions['minkowski_extension_sha256'] = sha256(spec.origin)
    return versions


def atomic_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def move_batch(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_batch(v, device) for k, v in batch.items()}
    return batch


class ResumeBatchSampler(Sampler):
    """Stateless epoch permutations; checkpoint cursor ignores DataLoader prefetch."""
    def __init__(self, size, batch_size, seed, start_batch, batches):
        if min(size, batch_size, batches) < 1 or start_batch < 0:
            raise ValueError('invalid batch sampler settings')
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.start, self.batches = start_batch, batches

    def __iter__(self):
        per_epoch = (self.size+self.batch_size-1)//self.batch_size
        previous, order = None, None
        for cursor in range(self.start, self.start+self.batches):
            epoch, index = divmod(cursor, per_epoch)
            if epoch != previous:
                generator = torch.Generator().manual_seed(self.seed+epoch)
                order = torch.randperm(self.size, generator=generator).tolist()
                previous = epoch
            yield order[index*self.batch_size:(index+1)*self.batch_size]

    def __len__(self):
        return self.batches


def plan(cfg):
    blockers, resolved = [], None
    try:
        hand = URDFHand(cfg['hand_spec'])
        resolved = asdict(model_config(cfg, hand.spec))
        if sha256(cfg['hand_spec']) != sha256(Path(cfg['dataset_root'])/'hand/hand_spec.json'):
            blockers.append('configured hand spec differs from dataset hand spec')
    except (ValueError, OSError, KeyError, TypeError) as exc:
        blockers.append(f'hand calibration: {exc}')
    try:
        dataset = audit_dataset(cfg['dataset_root'], cfg['model'].get('coord_scale_m', 0.2), deep=False)
    except (ValueError, OSError, KeyError, IndexError, TypeError) as exc:
        dataset = None
        blockers.append(f'dataset: {exc}')
    if importlib.util.find_spec('MinkowskiEngine') is None:
        blockers.append('MinkowskiEngine is not installed in this Python environment')
    else:
        try:
            import MinkowskiEngine as ME
            if torch.device(cfg['device']).type == 'cuda' and not ME.is_cuda_available():
                blockers.append('MinkowskiEngine was built CPU-only; this configuration requests CUDA')
        except (ImportError, OSError, RuntimeError) as exc:
            blockers.append(f'MinkowskiEngine extension cannot load: {exc}')
    if torch.device(cfg['device']).type == 'cuda' and not torch.cuda.is_available():
        blockers.append('CUDA unavailable in this Python environment')
    if cfg['method'] == 'flowmap' and not Path(cfg['teacher_checkpoint']).is_file():
        blockers.append('frozen FM teacher checkpoint is missing')
    if cfg.get('initialization') == 'checkpoint' and not Path(cfg['initial_checkpoint']).is_file():
        blockers.append('initial checkpoint is missing')
    return dict(status='blocked' if blockers else 'ready', training_started=False, blockers=blockers,
                config=cfg, model=resolved, dataset=dataset,
                effective_batch_size=cfg['batch_size']*cfg['accumulation'],
                evaluation_protocol='FM velocity MSE / FlowMap paired endpoint MSE are diagnostics; physical success is separate')


def load_model(checkpoint, device, *, encoder_factory=GeometryEncoder):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state.get('schema') != 'dexgrasp-v1' or state['method'] not in ('fm', 'flowmap'):
        raise ValueError('checkpoint is not a DexGrasp v1 model')
    model = KeypointModel(ModelConfig(**state['model_config']), state['method'], encoder_factory=encoder_factory).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    return model, state


def load_teacher(cfg, data_identity, hand_hash, device, encoder_factory):
    teacher, state = load_model(cfg['teacher_checkpoint'], device, encoder_factory=encoder_factory)
    if state['method'] != 'fm' or state.get('step', 0) < 1:
        raise ValueError('teacher must be an FM checkpoint with recorded optimizer steps')
    if state['data_identity'] != data_identity or state['hand_spec_sha256'] != hand_hash:
        raise ValueError('teacher and student must use the same dataset, split and hand calibration')
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


def autocast(cfg, device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=cfg['precision'] == 'bfloat16')


def initialize_student(student, teacher):
    """Copy FM parameters; initialize the new end-time input to zero, without sharing storage."""
    if student.method != 'flowmap' or teacher.method != 'fm':
        raise ValueError('expected FlowMap student and FM teacher')
    state = {k: v.detach().clone() for k, v in teacher.state_dict().items()}
    weight = state['time.0.weight']
    state['time.0.weight'] = torch.cat((weight, torch.zeros_like(weight)), dim=1)
    student.load_state_dict(state, strict=True)


def initialize_checkpoint(model, path, identity, hand_hash):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if (state.get('schema') != 'dexgrasp-v1' or state.get('method') != model.method
            or state.get('step', 0) < 1 or state['model_config'] != asdict(model.cfg)):
        raise ValueError('initial checkpoint method/dimensions mismatch or no training steps')
    if state['data_identity'] != identity or state['hand_spec_sha256'] != hand_hash:
        raise ValueError('initial checkpoint must use the same dataset and hand calibration')
    model.load_state_dict(state['model'], strict=True)
    return sha256(path)


def train_mode(model, cfg):
    model.train()
    if cfg.get('freeze_encoders', False):
        # Freezing parameters also freezes BatchNorm running statistics.
        model.object_encoder.eval()
        model.scene_encoder.eval()


@torch.no_grad()
def reconstruction_metrics(model, loader, cfg, device, *, seed=20000):
    """Paired MPJPE of generated endpoints; one random candidate per condition, no ranking."""
    distances = []
    devices = [device.index or 0] if device.type == 'cuda' else []
    with evaluation(model), torch.random.fork_rng(devices=devices):
        torch.manual_seed(cfg['seed']+seed)
        for batch in loader:
            batch = move_batch(batch, device)
            with autocast(cfg, device):
                predicted = sample_keypoints(model, batch['condition'], steps=cfg['sample_steps'])
            distance = (predicted-batch['keypoints_gt']).norm(dim=-1)*model.cfg.coord_scale_m*1000
            distances.append(distance.cpu())
    values = torch.cat(distances)
    return dict(mpjpe_mm=float(values.mean()), p95_keypoint_mm=float(torch.quantile(values.flatten(), .95)),
                max_sample_mpjpe_mm=float(values.mean(-1).max()))


@torch.no_grad()
def validate(model, loader, cfg, device):
    total, count = 0., 0
    devices = [device.index or 0] if device.type == 'cuda' else []
    with evaluation(model), torch.random.fork_rng(devices=devices):
        torch.manual_seed(cfg['seed']+10000)
        for i, batch in enumerate(loader):
            if i >= cfg['val_batches']:
                break
            batch = move_batch(batch, device)
            with autocast(cfg, device):
                if model.method == 'fm':
                    loss = fm_loss(model, batch)
                else:
                    # No held-out teacher trajectories are used as distillation supervision.
                    generated = sample_keypoints(model, batch['condition'], steps=cfg['sample_steps'])
                    loss = (generated-batch['keypoints_gt']).square().mean()
            n = len(batch['keypoints_gt'])
            total += float(loss)*n
            count += n
    if not count or not np.isfinite(total):
        raise ValueError('empty or nonfinite validation')
    return total/count


def run(cfg, *, resume=None, stop_after=None, encoder_factory=GeometryEncoder):
    hand = URDFHand(cfg['hand_spec'])
    hand_hash = sha256(cfg['hand_spec'])
    if hand_hash != sha256(Path(cfg['dataset_root'])/'hand/hand_spec.json'):
        raise ValueError('configured and dataset hand spec differ')
    mc = model_config(cfg, hand.spec)
    identity = audit_dataset(cfg['dataset_root'], mc.coord_scale_m, deep=True)
    identity['hand_urdf_sha256'] = sha256(hand.urdf_path)
    source = source_state()
    runtime = runtime_identity()
    device = torch.device(cfg['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if device.type == 'cuda' and cfg['precision'] == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 is unsupported on the selected CUDA device')
    run_dir, ckpt_dir = Path(cfg['run_dir']), Path(cfg['checkpoint_dir'])
    if run_dir == ckpt_dir:
        raise ValueError('run and checkpoint directories must differ')
    if not resume and any(p.exists() and any(p.iterdir()) for p in (run_dir, ckpt_dir)):
        raise FileExistsError('run/checkpoint directory is nonempty; use a new run ID or --resume')
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    model = KeypointModel(mc, cfg['method'], encoder_factory=encoder_factory).to(device)
    initial_hash = None
    if cfg.get('initialization') == 'checkpoint':
        initial_hash = initialize_checkpoint(model, cfg['initial_checkpoint'], identity, hand_hash)
    teacher = None
    teacher_hash = None
    if cfg['method'] == 'flowmap':
        teacher = load_teacher(cfg, identity, hand_hash, device, encoder_factory)
        if asdict(teacher.cfg) != asdict(mc):
            raise ValueError('v1 teacher/student must share the same model dimensions and geometry scales')
        teacher_hash = sha256(cfg['teacher_checkpoint'])
        if cfg.get('initialization', 'random') == 'fm_teacher':
            initialize_student(model, teacher)
    if cfg.get('freeze_encoders', False):
        model.object_encoder.requires_grad_(False)
        model.scene_encoder.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    start, best = 0, float('inf')
    if resume:
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if state.get('schema') != 'dexgrasp-v1' or state['config'] != cfg or state['data_identity'] != identity:
            raise ValueError('resume requires identical config/data identity')
        if state['source']['source_sha256'] != source['source_sha256'] or state['teacher_sha256'] != teacher_hash:
            raise ValueError('source or teacher changed; start a separate run')
        if state.get('initial_checkpoint_sha256') != initial_hash:
            raise ValueError('initial checkpoint changed; start a separate run')
        if state['runtime'] != runtime:
            raise ValueError('runtime/native extension changed; start a separate run')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start, best = state['step'], state['best_val']
        torch.set_rng_state(state['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    if start >= cfg['max_steps']:
        raise ValueError('checkpoint already reached max_steps')
    if stop_after is not None and not start < stop_after <= cfg['max_steps']:
        raise ValueError('stop_after must be after the resumed step and at most max_steps')
    datasets = {s: GraspDataset(cfg['dataset_root'], s, mc.coord_scale_m,
                               observation_cache_size=cfg['observation_cache_size'],
                               max_open_shards=cfg['max_open_shards']) for s in ('train', 'val')}
    sampler = ResumeBatchSampler(len(datasets['train']), cfg['batch_size'], cfg['seed'],
                                start*cfg['accumulation'], (cfg['max_steps']-start)*cfg['accumulation'])
    loader_options = dict(num_workers=cfg['num_workers'], pin_memory=device.type == 'cuda',
                          persistent_workers=cfg['num_workers'] > 0)
    # Separate RNGs avoid loader iterator creation consuming model noise/time RNG state.
    train_loader = DataLoader(datasets['train'], batch_sampler=sampler,
                              generator=torch.Generator().manual_seed(cfg['seed']), **loader_options)
    val_loader = DataLoader(datasets['val'], batch_size=cfg['batch_size'], shuffle=False,
                            generator=torch.Generator().manual_seed(cfg['seed']+1), **loader_options)
    diagnostic_loader = DataLoader(datasets['train'], batch_size=cfg['batch_size'], shuffle=False,
                                   generator=torch.Generator().manual_seed(cfg['seed']+2), **loader_options)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    val_name = 'velocity_mse' if cfg['method'] == 'fm' else 'paired_endpoint_mse_diagnostic'
    overfit = cfg.get('overfit_diagnostic', False)
    selection_metric = 'train_mpjpe_mm' if overfit else val_name
    if not resume:
        provenance = dict(config=cfg, model_config=asdict(mc), source=source, data_identity=identity,
                          hand_spec=hand.spec, teacher_sha256=teacher_hash, runtime=runtime,
                          validation_metric=val_name, checkpoint_selection_metric=selection_metric,
                          initialization=cfg.get('initialization', 'random'),
                          initial_checkpoint_sha256=initial_hash,
                          trainable_parameters=sum(p.numel() for p in trainable),
                          frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
                          frozen_teacher_parameters=sum(p.numel() for p in teacher.parameters()) if teacher else 0,
                          physical_success_evaluated=False,
                          teacher_supervision_split='train' if teacher is not None else None)
        with (run_dir/'provenance.json').open('x') as stream:
            json.dump(provenance, stream, indent=2, allow_nan=False)
    iterator = iter(train_loader)
    train_mode(model, cfg)
    started = time.monotonic()
    try:
        with (run_dir/'metrics.jsonl').open('a') as log:
            if overfit and not resume:
                initial = reconstruction_metrics(model, diagnostic_loader, cfg, device)
                log.write(json.dumps(dict(step=0, train_reconstruction=initial))+'\n')
                log.flush()
            for step in range(start+1, cfg['max_steps']+1):
                train_mode(model, cfg)
                optimizer.zero_grad(set_to_none=True)
                total = 0.
                for _ in range(cfg['accumulation']):
                    batch = move_batch(next(iterator), device)
                    with autocast(cfg, device):
                        if cfg['method'] == 'fm':
                            loss = fm_loss(model, batch)
                        else:
                            loss = flowmap_loss(model, teacher, batch, steps=cfg['teacher_steps'],
                                                solver=cfg['teacher_solver'],
                                                full_interval_probability=cfg['full_interval_probability'])
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite training loss')
                    (loss/cfg['accumulation']).backward()
                    total += float(loss.detach())/cfg['accumulation']
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
                optimizer.step()
                record = dict(step=step, loss=total, grad_norm=float(norm), seconds=time.monotonic()-started)
                improved = False
                if step % cfg['val_every'] == 0 or step == cfg['max_steps']:
                    value = validate(model, val_loader, cfg, device)
                    record[val_name] = value
                    if overfit:
                        record['train_reconstruction'] = reconstruction_metrics(model, diagnostic_loader, cfg, device)
                        value = record['train_reconstruction']['mpjpe_mm']
                    improved = value < best
                    best = min(best, value)
                log.write(json.dumps(record, allow_nan=False)+'\n')
                log.flush()
                if improved or step % cfg['save_every'] == 0 or step == cfg['max_steps'] or step == stop_after:
                    state = dict(schema='dexgrasp-v1', method=cfg['method'], step=step, config=cfg,
                                 model_config=asdict(mc), model=model.state_dict(), optimizer=optimizer.state_dict(),
                                 source=source, data_identity=identity, hand_spec_sha256=hand_hash,
                                 teacher_sha256=teacher_hash, best_val=best, validation_metric=val_name,
                                 checkpoint_selection_metric=selection_metric,
                                 initial_checkpoint_sha256=initial_hash,
                                 runtime=runtime,
                                 torch_rng=torch.get_rng_state(),
                                 cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
                    atomic_checkpoint(ckpt_dir/'last.pt', state)
                    if improved:
                        atomic_checkpoint(ckpt_dir/'best.pt', state)
                if step == stop_after:
                    break
    finally:
        for dataset in datasets.values():
            dataset.close()
    return dict(status='completed' if step == cfg['max_steps'] else 'paused', method=cfg['method'], steps=step, best_val=best,
                validation_metric=val_name, checkpoint_selection_metric=selection_metric,
                best_selection_value=best, physical_success_rate=None)
