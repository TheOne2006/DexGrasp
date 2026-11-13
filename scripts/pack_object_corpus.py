"""Verify raw object archives and publish portable mmap training tables atomically.

Creates an explicitly diagnostic object-hash train/test partition. It does not
freeze the research protocol or certify physical grasp success.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
from dexgrasp.data.dataset import contained,sha256
from dexgrasp.train.runner import source_state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--formal',action='store_true');p.add_argument('--seed',type=int,default=42)
    args=p.parse_args();source=Path(args.source).resolve();out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    index=json.loads((source/'INDEX.json').read_text())
    entries=[x for x in index['batches'] if x['kind'] in ('frozen_base','incremental')]
    if sum(x['kind']=='frozen_base' for x in entries)!=1:raise ValueError('one non-overlapping frozen base required')
    shards=[];observations=[];packages=[];source_identity={};hash_jobs=[]
    for e in entries:
        if e['kind']=='incremental' and e.get('disjoint_from_frozen_and_prior') is not True:raise ValueError('increment deduplication proof required')
        package=contained(source,e['path']);packages.append(package)
        if json.loads((package/'READY.json').read_text())['status']!='COMPLETE':raise ValueError('incomplete source')
        manifest=json.loads((package/'MANIFEST.json').read_text())
        source_identity[package.name]=dict(manifest_sha256=sha256(package/'MANIFEST.json'),ready_sha256=sha256(package/'READY.json'))
        for f in manifest['files']:
            path=contained(package,f['path']);hash_jobs.append((path,f))
            if f['path'].startswith('observations/') and f['path'].endswith('.npz'):
                observations.append((package,f['path']))
        for path in sorted((package/'receipts').glob('*.json')):
            r=json.loads(path.read_text())
            if r['output_count']:shards.append(dict(package=str(package),shard=r['shard'],count=r['output_count']))
    hand_identities=[(sha256(p/'hand/hand_spec.json'),sha256(p/'hand/diagnostic_hand.urdf')) for p in packages]
    if len(set(hand_identities))!=1:
        raise ValueError('Cannot combine packages with different hand calibration or kinematics')
    def verify(job):
        path,spec=job
        if path.stat().st_size!=spec['bytes'] or sha256(path)!=spec['sha256']:raise ValueError('source digest mismatch: '+str(path))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,_ in enumerate(pool.map(verify,hash_jobs)):
            if i%10000==0:print('verified',i,'/',len(hash_jobs),flush=True)
    verified_seconds=time.monotonic()-started
    formal_protocol = None
    if args.formal:
        from dexgrasp.data.formal_corpus import audit_and_split
        formal_test,formal_protocol = audit_and_split(shards,observations,out,packages[0]/'hand/hand_spec.json',args.seed)
        shutil.copytree(packages[0]/'hand',out/'hand')
    n=sum(s['count'] for s in shards);o=len(observations);arrays={};opened={}
    def create(name,shape,dtype):
        a=np.lib.format.open_memmap(out/(name+'.npy'),mode='w+',dtype=dtype,shape=shape)
        opened[name]=a;arrays[name]=dict(path=name+'.npy',shape=list(shape),dtype=a.dtype.str)
        return a
    points=create('object_points',(o,1024,3),'float32')
    target=create('keypoints_gt',(n,21,3),'float32');direction=create('grasp_direction_cmd',(n,3),'float32')
    gap=create('palm_gap',(n,1),'float32');obs_index=create('observation_index',(n,),'int64')
    centers=np.empty((o,3),np.float32);scales=np.empty((o,1),np.float32);lookup={}
    object_ids=[]
    for i,(package,relative) in enumerate(observations):
        with np.load(contained(package,relative),allow_pickle=False) as z:
            pc=z['points_m'];center=z['center_m'];scale=z['coord_scale_m']
            if pc.shape!=(1024,3) or not np.isfinite(pc).all() or not np.isfinite(center).all() or not np.all(scale==.2):raise ValueError('invalid observation geometry')
            points[i]=(pc-center[None])/scale[None];centers[i]=center;scales[i]=scale
            object_ids.append(str(z['object_id']))
        lookup[(str(package),relative)]=i
    point_seconds=time.monotonic()-started-verified_seconds
    test_object=formal_test if args.formal else {name:int(hashlib.sha256(('diagnostic-object-v1:'+name).encode()).hexdigest()[:8],16)%10==0 for name in set(object_ids)}
    test_mask=np.empty(n,dtype=bool);cursor=0
    for i,entry in enumerate(shards):
        package=Path(entry['package']);path=contained(package,entry['shard']);count=entry['count']
        with h5py.File(path,'r') as h:
            if (h.attrs.get('schema')!='sharpa-object-surface-meanh-v1' or h.attrs.get('direction_contract_version')!='official-hand-plus-z-v1' or h.attrs.get('palm_gap_contract_version')!='palm-surface-nearest-mean32-grid4y8z-m-v1' or not h.attrs.get('requires_palm_gap',False) or h.attrs.get('palm_sample_count')!=32 or len(h['K'])!=count):raise ValueError('source contract mismatch')
            names=h['observation'].asstr()[:];objects=h['object_id'].asstr()[:]
            oi=np.fromiter((lookup[(str(package),name)] for name in names),dtype=np.int64,count=count)
            if any(object_ids[j]!=name for j,name in zip(oi,objects)):raise ValueError('observation/object identity mismatch')
            k,d,hgap=h['K'][:],h['d'][:],h['h'][:]
            if k.shape!=(count,21,3) or d.shape!=(count,3) or hgap.shape!=(count,1) or not all(np.isfinite(a).all() for a in (k,d,hgap)):raise ValueError('invalid training labels')
            if not np.array_equal(d,h['R_source'][:,:,2]) or not np.array_equal(d,h['R'][:,:,1]) or not np.allclose(np.linalg.norm(d,axis=-1),1,atol=1e-5) or (hgap<0).any():raise ValueError('direction/gap contract mismatch')
            sl=slice(cursor,cursor+count)
            target[sl]=(k-centers[oi,None,:])/scales[oi,None,:];direction[sl]=d;gap[sl]=hgap;obs_index[sl]=oi
            test_mask[sl]=[test_object[name] for name in objects]
        cursor+=count
        if i%100==0:print('packed',i,'/',len(shards),'samples',cursor,flush=True)
    splits={}
    for split,mask in [('train',~test_mask),('test',test_mask)]:
        ids=np.flatnonzero(mask);a=create(split+'_rows',ids.shape,'int64');a[:]=ids
        splits[split]=dict(array=split+'_rows',count=len(ids))
    for a in opened.values():a.flush()
    for name,spec in arrays.items():spec['sha256']=sha256(out/spec['path']);spec['bytes']=(out/spec['path']).stat().st_size
    common=dict(coord_scale_m=.2,split_status='diagnostic_object_hash_not_research_frozen',source_identity=source_identity,
                hand_spec_sha256=hand_identities[0][0],hand_urdf_sha256=hand_identities[0][1],
                source_index_sha256=sha256(source/'INDEX.json'),object_count=len(test_object),samples=n,source=source_state())
    if args.formal:
        common.update(split_status='formal_object_asset_v1',split_protocol_sha256=sha256(out/'split_protocol.json'))
    manifest=dict(schema='dexgrasp-packed-v1',**common,arrays=arrays,splits=splits,
        condition={'object_points':dict(array='object_points',index='observation_index'),'grasp_direction_cmd':dict(array='grasp_direction_cmd'),'palm_gap':dict(array='palm_gap')},target=dict(array='keypoints_gt'))
    raw=dict(schema='dexgrasp-raw-index-v1',**common,shards=shards,splits={k:dict(path=arrays[v['array']]['path'],count=v['count']) for k,v in splits.items()})
    (out/'raw-index.json').write_text(json.dumps(raw,indent=2))
    (out/'packed.json.tmp').write_text(json.dumps(manifest,indent=2));os.replace(out/'packed.json.tmp',out/'packed.json')
    report=dict(total_seconds=time.monotonic()-started,source_verification_seconds=verified_seconds,observation_pack_seconds=point_seconds,source_files=len(hash_jobs),source_bytes=sum(f['bytes'] for _,f in hash_jobs),packed_bytes=sum(v['bytes'] for v in arrays.values()),samples=n,observations=o,splits=splits)
    (out/'packing-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
