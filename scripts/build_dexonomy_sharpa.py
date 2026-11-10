"""Build a source-backed Sharpa retargeting diagnostic; never assign physical success.

Dexonomy data: CC BY-NC 4.0. Retains archive/member hashes and source row indices.
A distant synthetic floor is added solely to supply the Data v1 environment channel.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_overfit_dataset import make_hand, box
from dexgrasp.data.retarget import ShadowLandmarks, retarget
from dexgrasp.data.prepare import ShardWriter
from dexgrasp.data.dataset import audit_dataset, sha256
from dexgrasp.geometry.hand import URDFHand
from dexgrasp.geometry.mesh import measure_palm_gap, sample_observation, validate_closed_mesh


def process_object(args):
    source, root, object_id, quota = args
    source, root = Path(source), Path(root)
    output = root/'candidates'/f'{object_id}.json'
    source_hash = sha256(root/'source/archive_members.json')
    inputs = [root/'source/archive_members.json', root/'source/shadow.xml', root/'hand/hand_spec.json', root/'hand/diagnostic_hand.urdf', Path(__file__), Path(__file__).parents[1]/'dexgrasp/data/retarget.py']
    fingerprint = hashlib.sha256(''.join(sha256(p) for p in inputs).encode()).hexdigest()
    if output.exists():
        cached = json.loads(output.read_text())
        if cached.get('cache_fingerprint') == fingerprint and all((root/'candidates'/f"{r['id']}.npz").exists() for r in cached['accepted']):
            return cached
    hand = URDFHand(root/'hand/hand_spec.json')
    shadow = ShadowLandmarks(root/'source/shadow.xml')
    paths = sorted(source.glob(f'Dexonomy_GRASP_shadow/succ_collect/*/{object_id}/floating/*.npy'))
    pool = []
    for p in paths:
        d = np.load(p, allow_pickle=True).item()
        scene = source/'DGN_5k/scene_cfg'/object_id/'floating'/p.name
        if not scene.exists():
            raise FileNotFoundError(scene)
        cfg = np.load(scene, allow_pickle=True).item()
        obj = cfg['scene'][cfg['task']['obj_name']]
        mesh_path = (scene.parent/obj['file_path']).resolve()
        if not mesh_path.is_relative_to(source.resolve()):
            raise ValueError('mesh outside selected source directory')
        # Validate serialized source scale against both upstream loaders.
        base_scale = np.asarray(obj['scale'], float)
        assert np.allclose(base_scale, float(p.stem[5:])/100)
        pose = np.asarray(obj['pose'], float)
        for i, q in enumerate(d['grasp_qpos']):
            pool.append((p, i, q, base_scale*float(d['scene_scale'][i]), pose, mesh_path))
    seed = int(hashlib.sha256(object_id.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    rng.shuffle(pool)
    meshes, accepted, rejects, seen = {}, [], Counter(), set()
    for path, index, src, scale, pose, mesh_path in pool:
        identity = hashlib.sha256(np.asarray(np.r_[src, scale], '<f4').tobytes()).hexdigest()
        if identity in seen:
            rejects['duplicate_source_pose'] += 1; continue
        seen.add(identity)
        if mesh_path not in meshes:
            mesh = trimesh.load(mesh_path, force='mesh', process=True)
            try:
                validate_closed_mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
            except ValueError:
                rejects['source_mesh_not_closed'] += 1; break
            meshes[mesh_path] = mesh
        mesh = meshes[mesh_path]
        v = (mesh.vertices*scale) @ Rotation.from_quat(pose[3:7], scalar_first=True).as_matrix().T + pose[:3]
        f = np.asarray(mesh.faces)
        result = retarget(hand, shadow, src)
        if not result['optimizer_success']:
            rejects['fit_not_converged'] += 1; continue
        if result['mpjpe_m'] > .015 or result['fingertip_max_m'] > .025:
            rejects['landmark_fit_error'] += 1; continue
        gap = measure_palm_gap(v, f, result['R'], result['t'], hand.spec['palm'], check_mesh=False)
        if not gap['valid']:
            rejects[gap['reason']] += 1; continue
        gid = identity[:20]
        # Supplementary skeleton/object penetration diagnostic, not mesh collision validation.
        skeleton = result['K']
        signed = trimesh.proximity.signed_distance(trimesh.Trimesh(v, f, process=False), skeleton)
        penetration = float(np.maximum(signed, 0).max())
        if penetration > .005:
            rejects['landmark_inside_object_over_5mm'] += 1; continue
        np.savez_compressed(root/'candidates'/f'{gid}.npz', vertices=v, faces=f,
                            source_qpos=src, source_scale=scale, **{k:result[k] for k in ('R','t','q','K','shadow_K')})
        accepted.append(dict(id=gid, object_id=object_id, source_member=str(path.relative_to(source)),
                             source_index=index, source_scene=str((source/'DGN_5k/scene_cfg'/object_id/'floating'/path.name).relative_to(source)),
                             source_mesh=str(mesh_path.relative_to(source)), scale=scale.tolist(),
                             mpjpe_mm=1000*result['mpjpe_m'], fingertip_max_mm=1000*result['fingertip_max_m'],
                             landmark_penetration_mm=1000*penetration, nfev=result['nfev'], **gap))
        if len(accepted)>=quota:break
    report = dict(object_id=object_id, pool=len(pool), accepted=accepted, rejects=dict(rejects), source_manifest_sha256=source_hash, cache_fingerprint=fingerprint)
    output.write_text(json.dumps(report, indent=2)+'\n')
    return report


def main(args):
    source, root = args.source.resolve(), args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not (root/'hand').exists():make_hand(args.sharpa.resolve(), root/'hand')
    spec_path=root/'hand/hand_spec.json'
    spec=json.loads(spec_path.read_text())
    spec['label_version']='dexonomy-shadow-to-sharpa-diagnostic-v1'
    spec_path.write_text(json.dumps(spec,indent=2)+'\n')
    for name in ('candidates','source','assets','scenes','raw_grasps'):(root/name).mkdir(exist_ok=True)
    shutil.copyfile(args.shadow, root/'source/shadow.xml')
    shutil.copyfile(args.license, root/'source/ShadowHand-Apache-2.0.txt')
    shutil.copyfile(source/'manifest.json', root/'source/archive_members.json')
    source_manifest=json.loads((source/'manifest.json').read_text())
    # Randomized deterministic object order; archive-prefix pool remains biased.
    objects=sorted(source_manifest['objects'])
    np.random.default_rng(20261110).shuffle(objects)
    args_pool=[(str(source),str(root),o,args.per_object) for o in objects]
    reports=[]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for r in executor.map(process_object,args_pool):
            reports.append(r)
            print(json.dumps(dict(object=r['object_id'],accepted=len(r['accepted']),pool=r['pool'],rejected=r['rejects'])),flush=True)
    eligible=[r for r in reports if len(r['accepted'])>=args.per_object]
    if len(eligible)<args.objects:
        raise ValueError(f'Only {len(eligible)} objects reached quota {args.per_object}; inspect candidates before changing protocol')
    chosen=eligible[:args.objects]
    if (root/'shards').exists():raise FileExistsError('dataset already assembled')
    (root/'shards').mkdir();(root/'splits').mkdir()
    hand=URDFHand(root/'hand/hand_spec.json')
    writer=ShardWriter(root/'shards/part-00000.h5',hand.spec_path)
    rows={s:[] for s in ('train','val','test')}; records=[]
    rng=np.random.default_rng(20261110)
    try:
        for j,report in enumerate(chosen):
            split='train' if j<16 else 'val' if j<18 else 'test'
            for record in report['accepted']:
                gid=record['id'];d=np.load(root/'candidates'/f'{gid}.npz',allow_pickle=False)
                target=(d['vertices'],d['faces'])
                # Shared fixed plane, well below the original floating object and hand.
                floor=box([1.,1.,.01],[0,0,-.40])
                if min(target[0][:,2].min(),d['K'][:,2].min()) < -.35:raise ValueError('floor too close')
                obs=sample_observation(target,floor,rng)
                scene_id='retarget_'+gid
                obs.update(object_id=record['object_id'],scene_id=scene_id,view_id='full_mesh')
                oi=writer.observation(obs)
                R,t,q=[d[k].astype(np.float32) for k in ('R','t','q')]
                K=hand.world_keypoints(R,t,q).astype(np.float32)
                row=dict(observation_index=oi,grasp_id=gid,R=R,t=t,q=q,K=K,d=R[:,2],
                         h=np.array([record['h']],np.float32),hit_ratio=np.float32(record['hit_ratio']))
                gi=writer.grasp(row);rows[split].append(dict(shard='shards/part-00000.h5',grasp_index=gi))
                record.update(split=split,grasp_index=gi,scene_id=scene_id)
                records.append(record)
                np.savez_compressed(root/f'assets/{gid}.npz',vertices=target[0],faces=target[1])
                np.savez_compressed(root/f'raw_grasps/{gid}.npz',R=R,t=t,q=q,K=K,shadow_K=d['shadow_K'],source_qpos=d['source_qpos'])
                scene=dict(scene_id=scene_id,object_id=record['object_id'],split=split,source=record,
                           environment='synthetic distant floor z=-0.4m; original source is floating',
                           physical_success=None,collision_validated=False,
                           label_source='bounded Sharpa IK retarget of Dexonomy Shadow landmarks',
                           mesh=f'assets/{gid}.npz')
                (root/f'scenes/{scene_id}.json').write_text(json.dumps(scene,indent=2)+'\n')
    finally:writer.close()
    for split, indices in rows.items():(root/f'splits/{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in indices))
    summary=dict(schema_version=1,dataset='sharpa_dexonomy_1k_v1',samples=len(records),objects=len(chosen),
                 split_counts={s:len(r) for s,r in rows.items()},seed=20261110,
                 source_pool_grasps=sum(r['pool'] for r in reports),
                 candidate_accepted=sum(len(r['accepted']) for r in reports),
                 rejection_counts=dict(sum((Counter(r['rejects']) for r in reports),Counter())),
                 source='Dexonomy / DGN meshes / Shadow successful-in-upstream-simulation grasps',
                 license='CC BY-NC 4.0; Sharpa URDF and Shadow model Apache-2.0',
                 selection='80 archive-prefix DGN objects; deterministic shuffled order; first 20 with 50 passing distinct rows each; not representative',
                 hand='Sharpa 22 DOF, project-defined 21 diagnostic landmarks',
                 scope='retargeted kinematic diagnostic; no Sharpa physical grasp success labels',
                 collision_validated=False,physical_success_rate=None,
                 geometry_checks=dict(joint_limits=True,closed_mesh=True,palm_hits_min=64,
                    fit_mean_max_mm=15,fit_fingertip_max_mm=25,landmark_penetration_max_mm=5,
                    full_hand_mesh_collision=False,self_collision=False),
                 median_retarget_mpjpe_mm=float(np.median([r['mpjpe_mm'] for r in records])),
                 p95_retarget_mpjpe_mm=float(np.quantile([r['mpjpe_mm'] for r in records],.95)),
                 audit=audit_dataset(root),builder_sha256=sha256(__file__),
                 retarget_sha256=sha256(Path(__file__).parents[1]/'dexgrasp/data/retarget.py'))
    (root/'records.json').write_text(json.dumps(records,indent=2)+'\n')
    (root/'preparation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--sharpa',type=Path,required=True);p.add_argument('--shadow',type=Path,required=True)
    p.add_argument('--license',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--per-object',type=int,default=50)
    p.add_argument('--objects',type=int,default=20)
    main(p.parse_args())
