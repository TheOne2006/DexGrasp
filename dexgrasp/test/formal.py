"""Frozen, object-balanced evaluation of formal packed Sharpa checkpoints."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import sqlite3
import time

import h5py
import numpy as np
import torch

from ..data.dataset import sha256, contained
from ..data.formal_corpus import verified_packed_identity
from ..geometry.direction import WRIST_Z, direction_definition
from ..geometry.hand import URDFHand
from ..train.runner import load_model, source_state, runtime_identity
from ..train.execution import configure_execution
from ..train.objectives import sample_keypoints
from .metrics import decode_metrics, summarize


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def official_hand(spec_path):
    """Override evaluator metadata only; never alter frozen training assets."""
    hand = URDFHand(spec_path)
    hand.spec = deepcopy(hand.spec)
    axis = np.asarray(hand.spec['official_from_grasp_rotation']).T @ [0., 0., 1.]
    hand.spec['direction_condition'] = dict(kind=WRIST_Z, local_axis=axis.tolist())
    direction_definition(hand.spec)
    return hand


def select_conditions(candidates, observations, directions, allowed_objects, *, count, seed):
    """Bottom-k seeded rows per object, deduplicating observation/direction pairs.

    candidates yields (global packed row, object_id). Its ordering does not
    affect selection. No model predictions or test scores participate.
    """
    if count < 1:
        raise ValueError('positive condition count required')
    selected = {obj: {} for obj in allowed_objects}
    for row, obj in candidates:
        if obj not in selected:
            continue
        key = (int(observations[row]), *np.asarray(directions[row]).round(6).tolist())
        score = hashlib.sha256(f'{seed}:{row}'.encode()).hexdigest()
        pool = selected[obj]
        if key in pool and pool[key][0] <= score:
            continue
        pool[key] = (score, int(row))
        if len(pool) > count:
            del pool[max(pool, key=lambda k: pool[k])]
    if any(not values for values in selected.values()):
        raise ValueError('selected object has no eligible conditions')
    return [(obj, row) for obj, values in sorted(selected.items())
            for _, row in sorted(values.values())]


def prepare(root, output, *, split='test', conditions=4, seed=7301, max_objects=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    identity = verified_packed_identity(root, .2)
    splits = json.loads((root/'split_protocol.json').read_text())
    objects = sorted(k for k, v in splits['object_splits'].items() if v == split)
    objects.sort(key=lambda obj: hashlib.sha256(f'{seed}:{obj}'.encode()).hexdigest())
    if max_objects is not None:
        if max_objects < 1:
            raise ValueError('positive object limit required')
        objects = objects[:max_objects]
    observations = np.load(root/'observation_index.npy', mmap_mode='r')
    directions = np.load(root/'grasp_direction_cmd.npy', mmap_mode='r')
    split_rows = np.load(root/f'{split}_rows.npy', mmap_mode='r')
    eligible = np.zeros(len(observations), dtype=bool)
    eligible[split_rows] = True
    # rowid follows insertion order in audit_and_split, exactly the shard order.
    # Verify each chosen row against raw object/id and packed target below.
    with sqlite3.connect(f'file:{root}/lineage.sqlite?mode=ro', uri=True) as con:
        candidates = ((int(rowid)-1, obj) for rowid, obj in con.execute('SELECT rowid, object FROM samples')
                      if eligible[int(rowid)-1])
        chosen = select_conditions(candidates, observations, directions, objects,
                                   count=conditions, seed=seed)
        ids = {row: con.execute('SELECT id FROM samples WHERE rowid=?', (row+1,)).fetchone()[0]
               for _, row in chosen}
    raw = json.loads((root/'raw-index.json').read_text())
    ends = np.cumsum([s['count'] for s in raw['shards']])
    packed_k = np.load(root/'keypoints_gt.npy', mmap_mode='r')
    points = np.load(root/'object_points.npy', mmap_mode='r')
    hand = official_hand(root/'hand/hand_spec.json')
    rows, hashes = [], {}
    by_shard = {}
    for obj, row in chosen:
        by_shard.setdefault(int(np.searchsorted(ends, row, side='right')), []).append((obj, row))
    for sid, selections in sorted(by_shard.items()):
        entry = raw['shards'][sid]
        package = Path(entry['package'])
        path = contained(package, entry['shard'])
        hashes[str(path)] = sha256(path)
        start = int(ends[sid-1]) if sid else 0
        with h5py.File(path, 'r') as h:
            if h.attrs.get('direction_contract_version') != 'official-hand-plus-z-v1':
                raise ValueError('unsupported source direction contract')
            for obj, row in selections:
                local = row-start
                if h['object_id'].asstr()[local] != obj or h['id'].asstr()[local] != ids[row]:
                    raise ValueError('raw/packed row identity mismatch')
                d, rotation = h['d'][local], h['R'][local]
                if not (np.allclose(d, rotation[:, 1], atol=1e-6) and
                        np.allclose(d, rotation @ hand.spec['direction_condition']['local_axis'], atol=1e-6)
                        and np.array_equal(d, directions[row])):
                    raise ValueError('official wrist +Z label mismatch')
                obs_path = contained(package, h['observation'].asstr()[local])
                with np.load(obs_path, allow_pickle=False) as z:
                    center = z['center_m']
                    if not np.allclose(points[observations[row]], (z['points_m']-center)/.2, atol=1e-6):
                        raise ValueError('observation normalization mismatch')
                k = h['K'][local]
                if not np.allclose(packed_k[row], (k-center)/.2, atol=1e-6):
                    raise ValueError('keypoint normalization mismatch')
                rows.append(dict(condition_id=ids[row], object_id=obj,
                                 asset_group=splits['object_groups'][obj], packed_row=row,
                                 observation_index=int(observations[row]), center_m=center.tolist(),
                                 direction=d.tolist(), source_shard=str(path), source_row=local,
                                 source_family=str(h['family'].asstr()[local]),
                                 gt_keypoints_centered_m=(k-center).tolist(), gt_q=h['q'][local].tolist()))
    rows.sort(key=lambda r: (r['object_id'], r['condition_id']))
    report = dict(schema='dexgrasp-formal-eval-selection-v1', dataset_root=str(root),
                  data_identity=identity, raw_index_sha256=sha256(root/'raw-index.json'),
                  split=split, seed=seed, conditions_per_object=conditions, objects=len(objects),
                  selection='seeded bottom-k unique observation/direction conditions per object; no score ranking',
                  direction_definition=hand.spec['direction_condition'],
                  coordinate_frame='object-centered metres; original object center retained per condition',
                  source_shard_sha256=hashes, rows=rows, source=source_state())
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(f'Selection: {len(objects)} objects, {len(rows)} conditions, {output}', flush=True)
    return report


_HAND = None


def init_worker(spec_path):
    global _HAND
    torch.set_num_threads(1)
    _HAND = official_hand(spec_path)


def decode_job(job):
    keypoints, direction, metadata = job
    return {**metadata, **decode_metrics(keypoints, _HAND, direction)}


def check_model_source(state):
    repo = Path(__file__).resolve().parents[2]
    hashes = state['source']['source_sha256']
    keys = [k for k in hashes if k.startswith(('dexgrasp/models/', 'dexgrasp/vendor/'))
            or k in ('dexgrasp/config.py', 'dexgrasp/train/objectives.py')]
    changed = [k for k in keys if sha256(repo/k) != hashes[k]]
    if changed:
        raise ValueError('checkpoint model source differs: '+', '.join(changed))


@torch.no_grad()
def run(checkpoint, selection, output, *, device='cuda:0', candidates=8,
        batch_size=8, workers=8, seed=8401, steps=20):
    checkpoint, selection, output = map(lambda p: Path(p).resolve(), (checkpoint, selection, output))
    from dexgrasp.test.checkpoint_policy import require_eligible
    require_eligible(checkpoint)
    if min(candidates, batch_size, workers, steps) < 1:
        raise ValueError('positive candidates, batch size, workers and sampling steps required')
    if output.exists():
        raise FileExistsError(output)
    chosen = json.loads(selection.read_text())
    root = Path(chosen['dataset_root'])
    if chosen.get('schema') != 'dexgrasp-formal-eval-selection-v1' or not chosen['rows']:
        raise ValueError('invalid selection manifest')
    identity = verified_packed_identity(root, .2)
    if identity != chosen['data_identity']:
        raise ValueError('selection dataset identity changed')
    # MinkowskiEngine uses the current CUDA device in sparse pooling kernels.
    # Moving tensors/model alone is insufficient for a nondefault GPU.
    if torch.device(device).type == 'cuda':
        torch.cuda.set_device(torch.device(device))
    model, state = load_model(checkpoint, torch.device(device))
    if state['data_identity'] != identity or state['method'] != 'fm':
        raise ValueError('checkpoint dataset/method mismatch')
    check_model_source(state)
    if model.cfg.use_gap or model.cfg.use_scene:
        raise ValueError('this evaluator requires object-only h-less checkpoints')
    if any(not torch.isfinite(t).all() for t in state['model'].values() if t.is_floating_point()):
        raise ValueError('nonfinite checkpoint tensors')
    cfg = dict(state['config'], compile_transformer=False)
    configure_execution(model, cfg)
    protocol = dict(schema='dexgrasp-formal-kinematic-evaluation-v1', checkpoint=str(checkpoint),
                    created_at=datetime.now(timezone.utc).isoformat(),
                    checkpoint_sha256=sha256(checkpoint), optimizer_steps=state['step'],
                    model_variant=model.cfg.variant, selection=str(selection), selection_sha256=sha256(selection),
                    data_identity=identity, source=source_state(), runtime=runtime_identity(),
                    training_source=state['source'], training_config=state['config'],
                    seed=seed, candidates=candidates, batch_size=batch_size, sampling_steps=steps,
                    sampler='Euler', precision='BF16 autocast; FP32 ODE accumulation', compile_transformer=False,
                    direction_definition=chosen['direction_definition'],
                    direction_metadata_override='evaluation-only official +Zh, verified against source R[:,1]',
                    coordinate_frame=chosen['coordinate_frame'],
                    diversity='mean sample std (ddof=1) of 22 joint angles within each fixed condition; no wrist pose; no physical-success filter',
                    aggregation='equal conditions within objects, then equal objects; 2000 asset-group bootstrap draws, seed 7301',
                    ik=dict(initialization='joint-limit midpoint; no GT q', max_nfev=100,
                            fit_success='max point L2 <= 5 mm AND joints in limits; optimizer status reported separately',
                            direction_tolerance_deg=10),
                    timing='CUDA synchronized; includes geometry encoder and Euler sampling; excludes H2D, warmup, loading and IK',
                    timing_warmup_batches=2, timing_device=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else 'CPU',
                    physical_success_evaluated=False, ranking='none; all failures retained')
    output.mkdir(parents=True)
    write_json(output/'protocol.json', protocol)
    del state
    rows = chosen['rows']
    points = np.load(root/'object_points.npy', mmap_mode='r')
    directions = np.load(root/'grasp_direction_cmd.npy', mmap_mode='r')
    noise = np.stack([torch.randn((candidates,21,3), generator=torch.Generator().manual_seed(
        seed+int(hashlib.sha256(r['condition_id'].encode()).hexdigest()[:12],16))).numpy() for r in rows])
    np.save(output/'noise.npy', noise)
    generated = np.lib.format.open_memmap(output/'keypoints_centered_m.npy', mode='w+', dtype='float32',
                                          shape=(len(rows),candidates,21,3))
    times, errors = [], []
    cuda = str(device).startswith('cuda')
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start+batch_size]
        condition = dict(object_points=torch.from_numpy(np.array(points[[r['observation_index'] for r in chunk]])).to(device),
                         grasp_direction_cmd=torch.from_numpy(np.array(directions[[r['packed_row'] for r in chunk]])).to(device))
        z = torch.from_numpy(noise[start:start+len(chunk)].reshape(-1,21,3)).to(device)
        if start == 0:
            for _ in range(2):
                try:
                    with torch.autocast(device_type='cuda' if cuda else 'cpu', dtype=torch.bfloat16, enabled=cuda):
                        sample_keypoints(model, condition, steps=steps, candidates=candidates, noise=z)
                except FloatingPointError:
                    pass
        if cuda:
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        try:
            with torch.autocast(device_type='cuda' if cuda else 'cpu', dtype=torch.bfloat16, enabled=cuda):
                result = sample_keypoints(model, condition, steps=steps, candidates=candidates, noise=z)
            if cuda:
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter()-tick
            generated[start:start+len(chunk)] = result.reshape(len(chunk),candidates,21,3).cpu().numpy()*.2
            times.append(dict(start=start, conditions=len(chunk), seconds=elapsed))
        except FloatingPointError as exc:
            # Recover finite candidates individually, rather than marking a
            # whole mixed batch invalid because one candidate failed.
            for ci in range(len(chunk)):
                one = {k:v[ci:ci+1] for k,v in condition.items()}
                for j in range(candidates):
                    try:
                        with torch.autocast(device_type='cuda' if cuda else 'cpu', dtype=torch.bfloat16, enabled=cuda):
                            pred = sample_keypoints(model, one, steps=steps, noise=z[ci*candidates+j:ci*candidates+j+1])
                        generated[start+ci,j] = pred[0].cpu().numpy()*.2
                    except FloatingPointError:
                        generated[start+ci,j] = np.nan
            errors.append(dict(start=start, error=str(exc)))
        if start % (batch_size*20) == 0:
            print(f'Generate {start+len(chunk)}/{len(rows)} conditions', flush=True)
    generated.flush()
    del model
    if cuda:
        torch.cuda.empty_cache()
    def jobs():
        for i, r in enumerate(rows):
            metadata = {k:r[k] for k in ('condition_id','object_id','asset_group','packed_row')}
            for j in range(candidates):
                yield generated[i,j], r['direction'], dict(**metadata, candidate=j)
    tick = time.perf_counter()
    decoded = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=init_worker, initargs=(str(root/'hand/hand_spec.json'),)) as pool:
        with (output/'candidates.jsonl').open('x') as log:
            for row in pool.map(decode_job, jobs(), chunksize=8):
                log.write(json.dumps(row, allow_nan=False)+'\n')
                decoded.append(row)
                if len(decoded) % 512 == 0:
                    log.flush()
                    print(f'IK {len(decoded)}/{len(rows)*candidates}, {time.perf_counter()-tick:.1f}s', flush=True)
    report = summarize(decoded)
    timed_candidates = sum(t['conditions'] for t in times)*candidates
    report['timing'] = dict(generation_ms_per_grasp=1000*sum(t['seconds'] for t in times)/timed_candidates if timed_candidates else None,
                            timed_candidate_fraction=timed_candidates/len(decoded), batches=times,
                            ik_mean_ms=1000*np.mean([r['ik_seconds'] for r in decoded]),
                            ik_wall_seconds=time.perf_counter()-tick, workers=workers,
                            generation_failures=errors)
    report['protocol_sha256'] = sha256(output/'protocol.json')
    write_json(output/'summary.json', report)
    print(json.dumps(dict(output=str(output), candidates=report['candidates'], macro=report['macro']), allow_nan=False), flush=True)
    return report


def compare(runs, output):
    """Export all prespecified runs; never rank/select a checkpoint on test."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    results, signatures, coverage = [], [], []
    for folder in map(Path, runs):
        protocol = json.loads((folder/'protocol.json').read_text())
        from dexgrasp.test.checkpoint_policy import exclusion
        if exclusion(protocol['checkpoint'], protocol['optimizer_steps']):
            continue
        summary = json.loads((folder/'summary.json').read_text())
        if summary['protocol_sha256'] != sha256(folder/'protocol.json'):
            raise ValueError('changed evaluation protocol: '+str(folder))
        selection = json.loads(Path(protocol['selection']).read_text())
        if sha256(protocol['selection']) != protocol['selection_sha256']:
            raise ValueError('changed evaluation selection')
        if summary['candidates'] != len(selection['rows'])*protocol['candidates']:
            raise ValueError('incomplete evaluation: '+str(folder))
        signatures.append([protocol[k] for k in ('selection_sha256','data_identity','seed','candidates',
                                                 'sampling_steps','sampler','precision','ik','direction_definition')]
                          +[sha256(folder/'noise.npy')])
        row = dict(run=folder.name, optimizer_steps=protocol['optimizer_steps'],
                   variant=protocol['model_variant'], objects=summary['objects'], candidates=summary['candidates'])
        for key in ('mean_keypoint_error_mm','mean_max_keypoint_error_mm','ik_fit_success_rate',
                    'direction_angular_error_deg','direction_pass_rate_10deg','joint_angle_diversity_rad',
                    'ik_valid_joint_angle_diversity_rad','decode_rate','finite_output_rate'):
            row[key] = summary['macro'][key]['mean']
        row['generation_ms_per_grasp'] = summary['timing']['generation_ms_per_grasp']
        row['ik_mean_ms'] = summary['timing']['ik_mean_ms']
        results.append(row)
        coverage.append(dict(run=folder.name, summary=str((folder/'summary.json').resolve()),
                             protocol_sha256=summary['protocol_sha256'],
                             summary_sha256=sha256(folder/'summary.json')))
    if not signatures or any(s != signatures[0] for s in signatures):
        raise ValueError('comparison requires identical selection/noise/sampling/data/metric protocols')
    output.mkdir(parents=True)
    write_json(output/'comparison.json', dict(rows=results, sources=coverage,
               note='All predefined checkpoints reported. Kinematic fit is not physical grasp success; no checkpoint selection on test.'))
    with (output/'comparison.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    print(json.dumps(results, indent=2), flush=True)
    return results
