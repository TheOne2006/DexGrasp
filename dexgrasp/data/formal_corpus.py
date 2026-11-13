"""Freeze object/asset/lineage groups and audit all labels before formal training."""
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np
import h5py
from .dataset import contained, sha256
from ..geometry.hand import URDFHand


def batched_fk(hand, q):
    q = np.asarray(q, dtype=np.float64)
    n = len(q)
    transforms = {hand.spec['root_link']: np.broadcast_to(np.eye(4), (n,4,4))}
    for parent, child, origin, axis, kind, index in hand.edges:
        frame = transforms[parent] @ origin
        if index is not None:
            motion = np.tile(np.eye(4), (n,1,1))
            if kind == 'revolute':
                x,y,z = axis
                cross = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
                angle = q[:,index,None,None]
                motion[:,:3,:3] = np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*(cross@cross)
            else:
                motion[:,:3,3] = q[:,index,None]*axis
            frame = frame @ motion
        transforms[child] = frame
    return np.stack([np.einsum('nij,j->ni', transforms[p['link']][:,:3,:3], p['local_xyz_m'])+
                     transforms[p['link']][:,:3,3] for p in hand.spec['keypoints']], axis=1)


class Groups:
    def __init__(self, names): self.parents = {name:name for name in names}
    def root(self, name):
        while self.parents[name] != name:
            self.parents[name] = self.parents[self.parents[name]]
            name = self.parents[name]
        return name
    def join(self, a, b):
        a,b = self.root(a),self.root(b)
        self.parents[max(a,b)] = min(a,b)


def audit_and_split(shards, observations, out, hand_path, seed):
    """Source checksums must have been verified by the caller before this audit."""
    names, assets = [], {}
    for package, relative in observations:
        with np.load(contained(package,relative), allow_pickle=False) as z:
            name, mesh = str(z['object_id']), str(z['source_mesh_sha256'])
        names.append(name); assets.setdefault(mesh,set()).add(name)
    groups = Groups(set(names))
    for objects in assets.values():
        first = min(objects)
        for name in objects: groups.join(first,name)
    hand = URDFHand(hand_path)
    database = Path(out)/'lineage.sqlite'
    con = sqlite3.connect(database)
    con.execute('PRAGMA journal_mode=OFF')
    con.execute('CREATE TABLE samples(id TEXT PRIMARY KEY, object TEXT NOT NULL, parent TEXT)')
    count = 0; max_fk_error = 0.; max_rotation_error = 0.
    try:
        for i,entry in enumerate(shards):
            with h5py.File(contained(entry['package'],entry['shard']),'r') as h:
                ids, obj, parents = (h[k].asstr()[:] for k in ('id','object_id','parent'))
                if len(ids)!=entry['count'] or any(name not in groups.parents for name in obj):
                    raise ValueError('invalid shard identity/count')
                con.executemany('INSERT INTO samples VALUES (?,?,?)', zip(ids,obj,parents))
                for begin in range(0,len(ids),4096):
                    sl = slice(begin,begin+4096)
                    q,r,t,k = (h[key][sl] for key in ('q','R','t','K'))
                    if (q.shape[1:]!=(22,) or r.shape[1:]!=(3,3) or t.shape[1:]!=(3,) or k.shape[1:]!=(21,3)
                            or not all(np.isfinite(v).all() for v in (q,r,t,k))):
                        raise ValueError('invalid finite FK labels')
                    if (q < hand.lower-1e-6).any() or (q > hand.upper+1e-6).any():
                        raise ValueError('joint limits violated')
                    orth = np.max(np.abs(r.transpose(0,2,1)@r-np.eye(3)))
                    max_rotation_error = max(max_rotation_error,float(orth))
                    if orth>1e-4 or not np.allclose(np.linalg.det(r),1,atol=1e-4):
                        raise ValueError('invalid rotations')
                    expected = batched_fk(hand,q)@r.transpose(0,2,1)+t[:,None,:]
                    error = float(np.max(np.abs(expected-k)))
                    max_fk_error = max(max_fk_error,error)
                    if error>2e-6: raise ValueError(f'FK mismatch {error}')
                count += len(ids)
            if i%100==0:
                con.commit();print('formal FK/identity',i,'/',len(shards),'rows',count,flush=True)
        con.commit()
        con.execute('CREATE INDEX parents ON samples(parent)')
        con.commit()
        # Parents are cross-object transfer templates, not object identities.
        # Record this dependence; never merge every object through the template bank.
        shared_parents = con.execute("SELECT count(*) FROM (SELECT parent FROM samples WHERE parent!='' GROUP BY parent HAVING count(DISTINCT object)>1)").fetchone()[0]
        mapping = {name:groups.root(name) for name in sorted(groups.parents)}
        test = {name:int(hashlib.sha256(f'dexgrasp-formal-v1:{seed}:{group}'.encode()).hexdigest()[:16],16)%10==0
                for name,group in mapping.items()}
        if not any(test.values()) or all(test.values()): raise ValueError('empty formal split')
        counts = {'train':0,'test':0}
        for name,n in con.execute('SELECT object,count(*) FROM samples GROUP BY object'):
            counts['test' if test[name] else 'train'] += n
        if count != sum(entry['count'] for entry in shards): raise ValueError('incomplete audit')
        con.execute('CREATE TABLE object_splits(object TEXT PRIMARY KEY, split TEXT)')
        con.executemany('INSERT INTO object_splits VALUES (?,?)', [(k,'test' if v else 'train') for k,v in test.items()])
        con.commit()
        cross_edges = con.execute('SELECT count(*) FROM samples s JOIN samples p ON s.parent=p.id JOIN object_splits a ON s.object=a.object JOIN object_splits b ON p.object=b.object WHERE a.split!=b.split').fetchone()[0]
        protocol = dict(schema='dexgrasp-formal-split-v1',seed=seed,
            rule='sha256(dexgrasp-formal-v1:seed:canonical_group)[:16] mod 10 == 0 is test',
            object_groups=mapping, object_splits={k:'test' if v else 'train' for k,v in test.items()},
            objects=len(mapping),groups=len(set(mapping.values())),samples=counts,
            audit=dict(rows=count,unique_ids=count,full_fk=True,max_fk_error_m=max_fk_error,
                       max_rotation_error=max_rotation_error,asset_groups=len(assets),
                       grouping='object_id + identical source_mesh_sha256',
                       shared_parent_templates=shared_parents,cross_split_parent_edges=cross_edges,
                       generalization='held-out objects/assets; NOT held-out generation lineages',
                       physical_success_evaluated=False))
        (Path(out)/'split_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        return test,protocol
    finally:
        con.close()


def verified_packed_identity(root, scale):
    root=Path(root)
    manifest=json.loads((root/'packed.json').read_text())
    protocol=json.loads((root/'split_protocol.json').read_text())
    if (manifest.get('split_status')!='formal_object_asset_v1' or
            manifest['coord_scale_m']!=scale or protocol['schema']!='dexgrasp-formal-split-v1' or
            not protocol['audit']['full_fk'] or protocol['audit']['rows']!=manifest['samples'] or
            manifest['split_protocol_sha256']!=sha256(root/'split_protocol.json')):
        raise ValueError('formal packed audit/split missing or changed')
    hand=URDFHand(root/'hand/hand_spec.json')
    if (sha256(hand.spec_path)!=manifest['hand_spec_sha256'] or
            sha256(hand.urdf_path)!=manifest['hand_urdf_sha256']):
        raise ValueError('formal packed hand identity changed')
    for name,spec in manifest['arrays'].items():
        path=contained(root,spec['path'])
        if path.stat().st_size!=spec['bytes'] or sha256(path)!=spec['sha256']:
            raise ValueError('packed content changed: '+name)
        a=np.load(path,mmap_mode='r',allow_pickle=False)
        if list(a.shape)!=spec['shape'] or a.dtype.str!=spec['dtype']:
            raise ValueError('packed shape/dtype changed: '+name)
    train=np.load(root/'train_rows.npy',mmap_mode='r')
    test=np.load(root/'test_rows.npy',mmap_mode='r')
    all_rows=np.sort(np.concatenate((train,test)))
    if not np.array_equal(all_rows,np.arange(manifest['samples'])):
        raise ValueError('split overlap or missing rows')
    if any(len(rows)!=protocol['samples'][key] or len(rows)!=manifest['splits'][key]['count']
           for key,rows in [('train',train),('test',test)]):
        raise ValueError('split counts changed')
    return dict(schema='dexgrasp-formal-packed-identity-v1',deep=True,samples=protocol['samples'],
                packed_sha256=sha256(root/'packed.json'),split_protocol_sha256=manifest['split_protocol_sha256'],
                hand_spec_sha256=manifest['hand_spec_sha256'],hand_urdf_sha256=manifest['hand_urdf_sha256'],
                source_identity=manifest['source_identity'],audit=protocol['audit'])
