"""Prepare assets and evaluate saved formal inference, without resampling or training."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
import time
import urllib.request
import xml.etree.ElementTree as ET

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from dexgrasp.geometry.hand import URDFHand
from dexgrasp.test.physics import PoseEvaluator, PROTOCOL, pca_diversity
from dexgrasp.test.checkpoint_policy import exclusion


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def fetch_hand(destination):
    """Keep original XML and licensing; derived XML removes visuals only."""
    destination = Path(destination)
    revision = 'd323b371d02f4fc04a0873c19c5234cae6d803dd'
    base = f'https://raw.githubusercontent.com/sharpa-robotics/sharpa-urdf-usd-xml/{revision}/'
    def download(pair):
        remote, local = pair
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.exists():
            with urllib.request.urlopen(base+remote, timeout=90) as response:
                data = response.read()
            local.write_bytes(data)
        return dict(url=base+remote, path=str(local), sha256=digest(local))
    manifest = [download(('wave_01/right_sharpa_wave/right_sharpa_wave.xml', destination/'original.xml'))]
    root = ET.parse(destination/'original.xml').getroot()
    for parent in root.iter():
        for geom in list(parent.findall('geom')):
            if geom.get('contype') == '0' and geom.get('conaffinity') == '0':
                parent.remove(geom)
    needed = {g.get('mesh') for g in root.iter('geom') if g.get('mesh')}
    asset = root.find('asset')
    downloads = [('LICENSE.txt', destination/'LICENSE.txt'), ('NOTICE.txt', destination/'NOTICE.txt')]
    for mesh in list(asset):
        if mesh.get('name') not in needed:
            asset.remove(mesh)
        else:
            downloads.append(('wave_01/right_sharpa_wave/meshes/'+mesh.get('file'),
                              destination/'meshes'/mesh.get('file')))
    with ThreadPoolExecutor(max_workers=6) as pool:
        manifest.extend(pool.map(download, downloads))
    ET.ElementTree(root).write(destination/'collision.xml', encoding='utf-8', xml_declaration=True)
    write(destination/'source.json', dict(repository=base, files=manifest,
          derived_xml_sha256=digest(destination/'collision.xml'), modifications='remove noncollision visual geoms and unused meshes only'))


DECOMPOSITION = dict(threshold=.03, preprocess_mode='auto', preprocess_resolution=50,
                     resolution=2000, mcts_nodes=20, mcts_iterations=100, mcts_max_depth=3,
                     max_convex_hull=32, seed=7301)


def decompose(job):
    source, destination = job
    destination = Path(destination)
    marker = destination/'manifest.json'
    if marker.exists():
        old = read(marker)
        if old['source_sha256'] != digest(source) or old['parameters'] != DECOMPOSITION:
            raise ValueError('decomposition cache identity changed')
        if any(digest(p) != sha for p, sha in old['pieces'].items()):
            raise ValueError('decomposition piece changed')
        return old
    import coacd
    import trimesh
    coacd.set_log_level('error')
    with np.load(source, allow_pickle=False) as z:
        vertices, faces = z['vertices'].astype(float), z['faces'].astype(np.int32)
    parts = coacd.run_coacd(coacd.Mesh(vertices, faces), **DECOMPOSITION)
    if not parts:
        raise ValueError('empty collision decomposition')
    destination.mkdir(parents=True, exist_ok=True)
    paths = {}
    for i, (v, f) in enumerate(parts):
        path = destination/f'{i:03d}.obj'
        trimesh.Trimesh(v, f, process=False).export(path)
        paths[str(path)] = digest(path)
    result = dict(source=str(source), source_sha256=digest(source), parameters=DECOMPOSITION, pieces=paths)
    write(marker, result)
    return result


def prepare(selection, assets, workers=8):
    selection, assets = Path(selection).resolve(), Path(assets).resolve()
    assets.mkdir(parents=True, exist_ok=True)
    fetch_hand(assets/'hand')
    chosen = read(selection)
    conditions, meshes = [], {}
    for index, row in enumerate(chosen['rows']):
        shard = Path(row['source_shard'])
        package = shard.parent.parent
        with h5py.File(shard, 'r') as h:
            i = row['source_row']
            if h['id'].asstr()[i] != row['condition_id'] or h['object_id'].asstr()[i] != row['object_id']:
                raise ValueError('source identity mismatch')
            obs_path = package/h['observation'].asstr()[i]
            gt = {key:h[key][i].tolist() for key in ('R', 't', 'q')}
        with np.load(obs_path, allow_pickle=False) as obs:
            source = package/str(obs['asset'])
            if not np.allclose(obs['center_m'], row['center_m'], atol=1e-8):
                raise ValueError('center mismatch')
            source_hash = digest(source)
            meshes[source_hash] = (str(source), str(assets/'objects'/source_hash))
            conditions.append(dict(index=index, condition_id=row['condition_id'], object_id=row['object_id'],
                source_mesh=str(source), source_mesh_sha256=source_hash, observation=str(obs_path),
                observation_sha256=digest(obs_path), scale=obs['scale'].tolist(), gt=gt,
                decomposition=str(assets/'objects'/source_hash/'manifest.json')))
    write(assets/'preparation.json', dict(selection=str(selection), selection_sha256=digest(selection),
                                         conditions=len(conditions), meshes=len(meshes), decomposition=DECOMPOSITION))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(decompose, meshes.values())):
            if i % 10 == 0:
                print('decomposed', i+1, '/', len(meshes), flush=True)
    write(assets/'conditions.json', dict(selection=str(selection), selection_sha256=digest(selection), conditions=conditions))
    print('ASSETS READY', len(conditions), flush=True)


_CONTEXT = None


def init_worker(context):
    global _CONTEXT
    _CONTEXT = context


def evaluate_condition(index):
    context = _CONTEXT
    target = Path(context['output'])/'chunks'/f'{index:05d}.json'
    if target.exists():
        previous = read(target)
        if previous['protocol_sha256'] != context['protocol_sha256'] or previous['index'] != index:
            raise ValueError('cached chunk identity mismatch')
        return index
    condition = context['conditions'][index]
    hand = URDFHand(context['hand_spec'])
    pieces = list(read(condition['decomposition'])['pieces'])
    evaluator = PoseEvaluator(hand, context['hand_xml'], pieces, condition['scale'])
    chosen = context['selection_rows'][index]
    gt = condition['gt']
    fk_error = evaluator.fk_error(gt['R'], gt['t'], gt['q'])
    if fk_error > .0001:
        raise ValueError(f'Sharpa XML/URDF FK mismatch {fk_error} m')
    result = dict(protocol_sha256=context['protocol_sha256'], index=index, fk_error_m=fk_error, rows=[])
    for name in [*context['runs'], 'ground_truth']:
        rows = context['runs'][name][index] if name != 'ground_truth' else [dict(
            condition_id=chosen['condition_id'], object_id=chosen['object_id'], asset_group=chosen['asset_group'],
            candidate=0, decoded=True, fit_success=True, q=gt['q'], R=gt['R'],
            t=(np.asarray(gt['t'])-chosen['center_m']).tolist())]
        for row in rows:
            out = {k:row[k] for k in ('condition_id', 'object_id', 'asset_group', 'candidate', 'decoded', 'fit_success')}
            out.update(run=name, geometry_evaluated=False, simulation_evaluated=False, single_pose_force_success=False)
            if row['decoded']:
                translation = np.asarray(row['t'])+chosen['center_m']
                out.update(evaluator.evaluate(row['R'], translation, row['q'], simulate=context['simulate']))
                official_rotation = np.asarray(row['R']) @ evaluator.official_from_grasp.T
                out['pose_features'] = np.r_[translation, Rotation.from_matrix(official_rotation).as_rotvec(), row['q']].tolist()
            else:
                out['failure_reason'] = 'decode_failed'
            result['rows'].append(out)
    write(target, result)
    return index


def summarize(output):
    output = Path(output)
    protocol = read(output/'protocol.json')
    chunks = [read(p) for p in sorted((output/'chunks').glob('*.json'))]
    if len(chunks) != protocol['conditions'] or [c['index'] for c in chunks] != list(range(protocol['conditions'])):
        raise ValueError('incomplete condition coverage')
    if any(c['protocol_sha256'] != digest(output/'protocol.json') for c in chunks):
        raise ValueError('protocol changed')
    all_rows = [r for c in chunks for r in c['rows']]
    metrics = ['hand_object_penetration_mm', 'self_penetration_mm', 'contact_distance_mm',
               'contact_distance_consistency_mm', 'contact_link_count']
    summaries = {}
    for name in [*protocol['inputs'], 'ground_truth']:
        rows = [r for r in all_rows if r['run'] == name]
        expected = protocol['conditions'] if name == 'ground_truth' else protocol['inputs'][name]['candidates']
        if len(rows) != expected or len({(r['condition_id'],r['candidate']) for r in rows}) != expected:
            raise ValueError('candidate coverage mismatch')
        objects = {}
        for r in rows:
            objects.setdefault(r['object_id'], []).append(r)
        per_object = []
        for obj, rs in sorted(objects.items()):
            data = dict(object_id=obj, asset_group=rs[0]['asset_group'], candidates=len(rs))
            # Frozen selection has equal candidates per condition. Keep explicit
            # condition means so reuse with unequal candidate counts stays safe.
            conditions = {}
            for r in rs:
                conditions.setdefault(r['condition_id'], []).append(r)
            for metric in metrics:
                means = [np.mean([r[metric] for r in cs if r.get(metric) is not None])
                         for cs in conditions.values() if any(r.get(metric) is not None for r in cs)]
                data[metric] = float(np.mean(means)) if means else None
                data[metric+'_coverage'] = sum(r.get(metric) is not None for r in rs)/len(rs)
            if protocol['simulate']:
                data['single_pose_gsr'] = float(np.mean([np.mean([r['single_pose_force_success'] for r in cs]) for cs in conditions.values()]))
                data['single_pose_osr'] = float(any(r['single_pose_force_success'] for r in rs))
                data['ik_and_single_pose_success_rate'] = float(np.mean([
                    r['fit_success'] and r['single_pose_force_success'] for r in rs]))
            data['collision_free_1mm_rate'] = float(np.mean([
                bool(r['geometry_evaluated'] and max(r['hand_object_penetration_mm'], r['self_penetration_mm']) <= 1)
                for r in rs]))
            data['hand_object_penetration_le_5mm_rate'] = float(np.mean([
                bool(r['geometry_evaluated'] and r['hand_object_penetration_mm'] <= 5) for r in rs]))
            data['finger_contact_rate_2mm'] = float(np.mean([
                bool(r['geometry_evaluated'] and r['contact_link_count'] > 0) for r in rs]))
            per_object.append(data)
        macro = {}
        groups = sorted({r['asset_group'] for r in per_object})
        draws = np.random.default_rng(7301).integers(len(groups), size=(2000, len(groups)))
        for key in per_object[0]:
            if key in ('object_id', 'asset_group', 'candidates'):
                continue
            values = [r[key] for r in per_object if r[key] is not None]
            sums = np.array([sum(r[key] for r in per_object if r['asset_group']==g and r[key] is not None) for g in groups])
            counts = np.array([sum(r['asset_group']==g and r[key] is not None for r in per_object) for g in groups])
            denominator = counts[draws].sum(1)
            valid = denominator > 0
            estimates = sums[draws].sum(1)[valid]/denominator[valid]
            macro[key] = dict(mean=float(np.mean(values)) if values else None,
                              ci95=np.quantile(estimates, [.025,.975]).tolist() if len(estimates) else None)
        decoded = [r for r in rows if 'pose_features' in r]
        successful = [r for r in decoded if r['single_pose_force_success']]
        summary = dict(candidates=len(rows), objects=len(objects), macro=macro, per_object=per_object,
            geometry_evaluated=sum(r['geometry_evaluated'] for r in rows),
            simulation_evaluated=sum(r['simulation_evaluated'] for r in rows),
            pca_cumulative_variance_pct_all=pca_diversity([r['pose_features'] for r in decoded]),
            pca_cumulative_variance_pct_success=pca_diversity([r['pose_features'] for r in successful]),
            successful_candidates=len(successful) if protocol['simulate'] else None,
            failure_reasons=dict(Counter(r.get('failure_reason','passed' if r['single_pose_force_success'] else 'not_simulated') for r in rows)),
            distributions={k:dict(count=sum(k in r for r in rows),
                p50_p90_p95=np.quantile([r[k] for r in rows if k in r], [.5,.9,.95]).tolist()
                if any(k in r for r in rows) else None) for k in metrics})
        if name != 'ground_truth':
            old = read(Path(protocol['inputs'][name]['path'])/'summary.json')
            summary['kinematic_macro'] = old['macro']
            summary['inference_timing'] = {k:v for k,v in old.get('timing',{}).items() if k != 'batches'}
        summaries[name] = summary
    write(output/'summary.json', dict(protocol_sha256=digest(output/'protocol.json'), runs=summaries,
          max_fk_error_m=max(c['fk_error_m'] for c in chunks), note=PROTOCOL['comparability']))
    import csv
    with (output/'comparison.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['run','candidates','objects',*summaries[next(iter(summaries))]['macro'],
                         'pca_pc1_all_pct','pca_pc1_success_pct'])
        for name,s in summaries.items():
            writer.writerow([name,s['candidates'],s['objects'],*[m['mean'] for m in s['macro'].values()],
                (s['pca_cumulative_variance_pct_all'] or [None])[0],
                (s['pca_cumulative_variance_pct_success'] or [None])[0]])
    with (output/'candidates.jsonl').open('w') as f:
        for r in all_rows:
            f.write(json.dumps(r, allow_nan=False)+'\n')
    print(json.dumps({n:{k:v['mean'] for k,v in s['macro'].items() if not k.endswith('coverage')}
                      for n,s in summaries.items()}, indent=2), flush=True)


def run(selection, assets, runs, output, workers=16, geometry_only=False, limit=None):
    eligible, excluded = [], []
    for folder in runs:
        p = read(Path(folder)/'protocol.json')
        reason = exclusion(p['checkpoint'], p['optimizer_steps'])
        if reason:
            excluded.append(dict(run=str(folder), **reason))
            print('SKIP excluded checkpoint:', folder, flush=True)
        else:
            eligible.append(folder)
    runs = eligible
    if not runs:
        raise ValueError('No eligible checkpoints remain for physics evaluation')
    selection, assets, output = map(lambda p:Path(p).resolve(), (selection,assets,output))
    chosen = read(selection)
    asset_info = read(assets/'conditions.json')
    hand_source = read(assets/'hand/source.json')
    if (any(digest(f['path']) != f['sha256'] for f in hand_source['files']) or
            digest(assets/'hand/collision.xml') != hand_source['derived_xml_sha256']):
        raise ValueError('official hand assets changed since preparation')
    if asset_info['selection_sha256'] != digest(selection):
        raise ValueError('asset/selection mismatch')
    count = len(chosen['rows']) if limit is None else min(limit, len(chosen['rows']))
    if count < 1 or workers < 1:
        raise ValueError('positive limit and workers required')
    for c in asset_info['conditions'][:count]:
        if (digest(c['source_mesh']) != c['source_mesh_sha256'] or
                digest(c['observation']) != c['observation_sha256']):
            raise ValueError('archived geometry/observation changed')
        if any(digest(p) != sha for p,sha in read(c['decomposition'])['pieces'].items()):
            raise ValueError('cached convex pieces changed')
    inputs, candidates = {}, {}
    for folder in runs:
        folder = Path(folder).resolve()
        p, s = read(folder/'protocol.json'), read(folder/'summary.json')
        if p['selection_sha256'] != digest(selection) or s['protocol_sha256'] != digest(folder/'protocol.json'):
            raise ValueError('inference protocol mismatch')
        name = folder.name
        if name in inputs or name == 'ground_truth':
            raise ValueError('duplicate run name')
        rows = [json.loads(l) for l in (folder/'candidates.jsonl').open()]
        n = p['candidates']
        if len(rows) != len(chosen['rows'])*n or len(rows) != s['candidates']:
            raise ValueError('incomplete inference')
        for i, r in enumerate(rows):
            selected = chosen['rows'][i//n]
            if r['candidate'] != i % n or any(r[k] != selected[k] for k in ('condition_id','object_id','asset_group','packed_row')):
                raise ValueError('inference candidate order mismatch')
        candidates[name] = [rows[i*n:(i+1)*n] for i in range(count)]
        inputs[name] = dict(path=str(folder), checkpoint_sha256=p['checkpoint_sha256'],
            candidates=count*n, protocol_sha256=digest(folder/'protocol.json'),
            candidates_sha256=digest(folder/'candidates.jsonl'), summary_sha256=digest(folder/'summary.json'))
    import importlib.metadata
    repo = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__),repo/'dexgrasp/test/physics.py',repo/'dexgrasp/test/checkpoint_policy.py',repo/'dexgrasp/geometry/hand.py',
                    repo/'dexgrasp/geometry/direction.py',repo/'dexgrasp/__init__.py',
                    repo/'dexgrasp/test/__init__.py',repo/'dexgrasp/geometry/__init__.py']
    asset_paths = [assets/'conditions.json', *sorted((assets/'hand').rglob('*'))]
    for c in asset_info['conditions'][:count]:
        asset_paths.extend([Path(c['decomposition']), *map(Path, read(c['decomposition'])['pieces'])])
    protocol = dict(**PROTOCOL, inputs=inputs, excluded_runs=excluded, selection=str(selection), selection_sha256=digest(selection),
        split=chosen['split'], data_identity=chosen['data_identity'], conditions=count,
        full_selection=count==len(chosen['rows']), simulate=not geometry_only,
        source_sha256={str(p.relative_to(repo)):digest(p) for p in source_paths},
        asset_sha256={str(p):digest(p) for p in set(asset_paths) if p.is_file()},
        hand_spec=str(Path(chosen['dataset_root'])/'hand/hand_spec.json'),
        hand_spec_sha256=digest(Path(chosen['dataset_root'])/'hand/hand_spec.json'),
        hand_urdf_sha256=digest(URDFHand(Path(chosen['dataset_root'])/'hand/hand_spec.json').urdf_path),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        runtime={k:importlib.metadata.version(k) for k in ('mujoco','numpy','scipy','coacd')},
        aggregation='equal conditions within object; equal objects; asset-group bootstrap 2000, seed 7301',
        diversity='PCA unstandardized original-object t(m), official wrist rotvec(rad), q(rad); all decoded and successful separately')
    if (output/'protocol.json').exists():
        if read(output/'protocol.json') != protocol:
            raise ValueError('resume protocol/input/source mismatch; choose a new output')
    else:
        output.mkdir(parents=True, exist_ok=False)
        write(output/'protocol.json', protocol)
        for p in source_paths:
            dest = output/'source'/p.relative_to(repo)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
    lock = (output/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    context = dict(output=str(output), conditions=asset_info['conditions'], selection_rows=chosen['rows'],
                   runs=candidates, hand_spec=protocol['hand_spec'], hand_xml=str(assets/'hand/collision.xml'),
                   simulate=not geometry_only, protocol_sha256=digest(output/'protocol.json'))
    started = time.time()
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(context,)) as pool:
            for i, index in enumerate(pool.map(evaluate_condition, range(count))):
                if i % 10 == 0 or i+1 == count:
                    state = dict(completed_conditions=i+1, total_conditions=count, elapsed_seconds=time.time()-started,
                                 status='running', pid=os.getpid())
                    write(output/'status.json',state)
                    print(state,flush=True)
    except Exception as exc:
        write(output/'status.json',dict(status='failed', error=str(exc), total_conditions=count,
              completed_conditions=len(list((output/'chunks').glob('*.json')))))
        raise
    summarize(output)
    write(output/'status.json',dict(status='completed', completed_conditions=count, total_conditions=count,
                                   elapsed_seconds=time.time()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--selection', required=True); p.add_argument('--assets',required=True)
    p.add_argument('--workers',type=int,default=8)
    p = sub.add_parser('run')
    for name in ('selection','assets','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--runs',nargs='+',required=True); p.add_argument('--workers',type=int,default=16)
    p.add_argument('--geometry-only',action='store_true'); p.add_argument('--limit',type=int)
    p = sub.add_parser('summarize'); p.add_argument('--output',required=True)
    args = vars(parser.parse_args()); action = args.pop('action')
    globals()[action](**args)
