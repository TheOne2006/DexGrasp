"""Table-preserving Slide -> six independent FC branches (custom protocol v1).

Imports are CPU-only. The Warp backend is loaded only by an explicitly selected
runtime. Physical tests and safety checks are never gated by engineering audit.
"""
from __future__ import annotations
import copy
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
import mujoco
import numpy as np
from dexonomy.sim import HandCfg, MuJoCo_EvalCfg, MuJoCo_EvalEnv
from dexonomy.sim.mujoco_env import physics_stage_arena_memory_bytes
from dexonomy.util.file_util import load_scene_cfg
from dexonomy.util.np_util import np_get_delta_pose
from dexonomy.op.eval_force_closure_warp import _interpolate_segment
from dexonomy.tabletop_trajectory import SlideCfg, SlidePlanner
from dexonomy.sim.fc_execution import direction_tasks, add_direction, ordered_directions

SCHEMA = 'dexonomy-tabletop-slide-state/v1'
DIRECTIONS = np.asarray([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],float)
# Integration inputs plus derived fields used before the first subsequent step.
STATE = ('time','qpos','qvel','act','history','qacc_warmstart','ctrl','qfrc_applied',
         'xfrc_applied','eq_active','mocap_pos','mocap_quat','userdata','qacc','act_dot',
         'xpos','xquat','xmat','xipos','ximat','cdof','subtree_com')


def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)


def digest(value):
    def typed(x):
        if isinstance(x,np.ndarray):return {'dtype':str(x.dtype),'shape':x.shape,'values':x.tolist()}
        return serial(x)
    return hashlib.sha256(json.dumps(value,default=typed,sort_keys=True,allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Protocol:
    trans_threshold: float = .05
    rot_threshold_deg: float = 15.
    table_contact_tolerance_m: float = 0.0
    holds: int = 10
    steps_per_hold: int = 50
    noslip_iterations: int = 0

    def __post_init__(self):
        if (self.trans_threshold,self.rot_threshold_deg,self.holds,self.steps_per_hold,self.noslip_iterations)!=(.05,15.,10,50,0):
            raise ValueError('v1 preserves fixed official evaluation parameters; Eval noslip0')
        if not np.isfinite(self.table_contact_tolerance_m) or self.table_contact_tolerance_m<0:
            raise ValueError('nonnegative geometric contact tolerance required')


def make_host(scene_path, hand_xml):
    scene=load_scene_cfg(str(scene_path))
    task=scene['task']
    if task['type']!='slide' or task['distance']!=.1 or not np.array_equal(task['axis'],[0,0,1]):
        raise ValueError('requires official vertical 0.1m Slide scene')
    planes=[x for x in scene['scene'].values() if x['type']=='plane']
    objects=[x for x in scene['scene'].values() if x['type']=='rigid_object']
    if len(planes)!=1 or len(objects)!=1 or len(scene['scene'])!=2:
        raise ValueError('v1 supports one rigid target and one world plane')
    obj=scene['scene'][task['obj_name']]
    if obj.get('fix_root',False):raise ValueError('Slide target must be movable')
    info=json.loads(Path(obj['info_path']).read_text())
    coef=info['mass']/(info['density']*info['scale']**3)
    obj['density']=.1/(coef*float(np.prod(obj['scale'])))
    host=MuJoCo_EvalEnv(HandCfg(str(hand_xml),freejoint=True),scene,
        MuJoCo_EvalCfg(miu_coef=[.6,.02],noslip_iterations=0,physics_profile='native',
                      arena_memory_bytes=physics_stage_arena_memory_bytes()),False,False)
    host._model.opt.tolerance=1e-8;host._model.opt.noslip_tolerance=1e-6
    if host._model.nplugin or host._model.nuserdata:
        raise ValueError('v1 excludes plugin/controller userdata state')
    host.tabletop_scene=scene
    host.tabletop_identity={'scene_path':str(Path(scene_path).resolve()),'scene_sha256':hashlib.sha256(Path(scene_path).read_bytes()).hexdigest(),
        'scene_id':scene.get('scene_id',str(scene_path)),'scale':np.asarray(obj['scale']).tolist(),'pose':np.asarray(obj['pose']).tolist(),
        'mujoco':mujoco.__version__,'protocol':'tabletop-slide-fc/v1','model_identity_schema':'public-model-arrays-options/v2'}
    host.tabletop_identity['model_sha256']=model_identity(host._model)
    host.tabletop_categories=geom_categories(host)
    return host


def model_identity(model):
    h=hashlib.sha256()
    for owner,prefix in ((model,''),(model.opt,'opt.')):
        for name in sorted(dir(owner)):
            # Native allocation addresses are process-local, not model identity.
            if name.startswith('_'):continue
            value=getattr(owner,name)
            if isinstance(value,np.ndarray):
                h.update((prefix+name+str(value.dtype)+str(value.shape)).encode());h.update(value.tobytes())
            elif isinstance(value,(float,int)):
                h.update((prefix+name+repr(value)).encode())
    return h.hexdigest()


def is_descendant(model,body,root):
    while body>0 and body!=root:body=int(model.body_parentid[body])
    return body==root


def geom_categories(host):
    model=host._model;result=np.zeros(model.ngeom,np.int32)
    for i in range(model.ngeom):
        if not (model.geom_contype[i] or model.geom_conaffinity[i]):continue
        name=model.geom(i).name;body=int(model.geom_bodyid[i])
        if name.startswith(host._cfg.plane_prefix) and model.geom_type[i]==mujoco.mjtGeom.mjGEOM_PLANE:result[i]=1
        elif is_descendant(model,body,host._obj_active_bodyid):result[i]=2
        elif model.body(body).name.startswith(host._cfg.hand_prefix):result[i]=3
    if not np.any(result==1) or not np.any(result==2) or not np.any(result==3):
        raise ValueError('table/target/hand collision inventory incomplete')
    return result


def contact_summary(categories, pairs, distances, tolerance):
    flags=np.zeros(2,np.int32);near=np.zeros(2,np.int32);minimum=np.full(2,np.inf)
    for pair,dist in zip(pairs,distances):
        if np.any(np.asarray(pair)<0) or np.any(np.asarray(pair)>=len(categories)):raise ValueError('invalid contact geom')
        a,b=categories[pair[0]],categories[pair[1]]
        other=b if a==1 else a if b==1 else 0
        if other not in (2,3):continue
        if not np.isfinite(dist):raise ValueError('non-finite contact distance')
        j=other-2;minimum[j]=min(minimum[j],dist)
        if dist<=tolerance:flags[j]+=1
        else:near[j]+=1
    return flags,near,minimum


def snapshot(identity,candidate_id,backend,fields,hold_target,table_flags):
    copied={k:np.array(v,copy=True) for k,v in fields.items()}
    if set(copied)!=set(STATE):raise ValueError('incomplete continuation state')
    for v in copied.values():
        if not np.isfinite(v).all():raise ValueError('non-finite snapshot')
    result={'schema':SCHEMA,'identity':copy.deepcopy(identity),'candidate_id':candidate_id,'backend':backend,
            'fields':copied,'hold_target':np.array(hold_target,copy=True),'initial_table_contacts':np.array(table_flags,copy=True),
            'restore_policy':'restore all STATE fields; clear applied forces; no preamble/reset/forward; next step rebuilds contact and solver workspaces'}
    result['sha256']=digest(result)
    for value in copied.values():value.setflags(write=False)
    return result


def check_snapshot(saved,host,candidate_id,backend):
    if saved['schema']!=SCHEMA or saved['candidate_id']!=candidate_id or saved['backend']!=backend or saved['identity']!=host.tabletop_identity:
        raise ValueError('snapshot schema/candidate/backend/scene/model identity mismatch')
    if saved['sha256']!=digest({k:v for k,v in saved.items() if k!='sha256'}):raise ValueError('snapshot content mismatch')
    if set(saved['fields'])!=set(STATE):raise ValueError('missing continuation field')


class CpuBatch:
    backend='mujoco-3.17'
    def __init__(self,hosts,indices,starts=None,snapshots=None,candidate_ids=None):
        self.envs=[];self.hosts=[hosts[i] for i in indices]
        for i,host in enumerate(self.hosts):
            env=copy.copy(host);env._data=mujoco.MjData(host._model)
            if snapshots is None:
                q=np.asarray(starts[i]);env._data.qpos[:]=np.concatenate([q,env._obj_init_qpos]);env.set_ctrl(q)
                mujoco.mj_forward(env._model,env._data)
            else:
                saved=snapshots[i];check_snapshot(saved,host,candidate_ids[i],self.backend)
                for name in STATE:
                    value=saved['fields'][name];dest=getattr(env._data,name)
                    if np.shape(dest)!=value.shape:raise ValueError('snapshot field shape: '+name)
                    if name=='time':env._data.time=float(value)
                    else:dest[:]=value
            env._data.qfrc_applied[:]=0;env._data.xfrc_applied[:]=0
            self.envs.append(env)
        self.flags=np.zeros((len(indices),2),np.int64);self.near=self.flags.copy();self.minimum=np.full((len(indices),2),np.inf)
    def control(self,targets):
        for e,q in zip(self.envs,targets):e.set_ctrl(q)
    def force(self,directions):
        for e,f in zip(self.envs,directions):e.set_active_obj_ext_fdir(np.asarray(f))
    def step(self,n,contact=False,tolerance=0.):
        for _ in range(n):
            for i,e in enumerate(self.envs):
                mujoco.mj_step(e._model,e._data)
                if not np.isfinite(e._data.qpos).all() or not np.isfinite(e._data.qvel).all():raise ValueError('non-finite simulation')
                if e._data.warning.number.any():raise RuntimeError('MuJoCo warning: '+str(e._data.warning.number))
                if contact:
                    d=e._data;f,near,m=contact_summary(e.tabletop_categories,d.contact.geom[:d.ncon],d.contact.dist[:d.ncon],tolerance)
                    self.flags[i]+=f;self.near[i]+=near;self.minimum[i]=np.minimum(self.minimum[i],m)
    def poses(self):return np.stack([e.get_active_obj_pose() for e in self.envs])
    def qpos(self):return np.stack([e._data.qpos.copy() for e in self.envs])
    def states(self):return [{k:np.array(getattr(e._data,k),copy=True) for k in STATE} for e in self.envs]
    def query_contacts(self,tolerance):
        rows=[]
        for e in self.envs:
            # Contact at the integrated endpoint, without mutating continuation.
            query=mujoco.MjData(e._model)
            spec=mujoco.mjtState.mjSTATE_INTEGRATION
            state=np.empty(mujoco.mj_stateSize(e._model,spec))
            mujoco.mj_getState(e._model,e._data,state,spec);mujoco.mj_setState(e._model,query,state,spec)
            mujoco.mj_forward(e._model,query)
            rows.append(contact_summary(e.tabletop_categories,query.contact.geom[:query.ncon],query.contact.dist[:query.ncon],tolerance))
        return rows
    def end_contacts(self,tolerance):return np.stack([r[0] for r in self.query_contacts(tolerance)])
    def contacts(self,tolerance=0.):
        for i,(flags,near,minimum) in enumerate(self.query_contacts(tolerance)):
            self.flags[i]+=flags;self.near[i]+=near;self.minimum[i]=np.minimum(self.minimum[i],minimum)
        return self.flags.copy(),self.near.copy(),self.minimum.copy()


def slide_plan(host,data):
    task=dict(host.tabletop_scene['task']);task['part_name']=task.get('part_name')
    planner=SlidePlanner(SlideCfg(**task))
    controls,types,forces,target=planner.plan_trajectory(host.get_active_obj_pose().copy(),np.asarray(data['pregrasp_qpos']),
        np.asarray(data['grasp_qpos']),np.asarray(data['squeeze_qpos']),data.get('robot_pose'))
    if any(t!='ee_pose' for t in types):raise ValueError('v1 supports free-hand ee_pose controls only')
    segments=[_interpolate_segment(a,b,types[i],5) for i,(a,b) in enumerate(zip(controls[:-1],controls[1:]))]
    if len(forces)<len(segments) or any(x is None for x in forces[:len(segments)]):raise ValueError('Slide force plan incomplete')
    return {'start':controls[0],'segments':segments,'forces':forces[:len(segments)],'target':target,'hold_target':controls[-1]}


def metrics(target,pose,policy):
    if not np.isfinite(pose).all():raise ValueError('non-finite object pose')
    trans,rot=np_get_delta_pose(target,pose)
    return {'translation_m':float(trans),'rotation_deg':float(rot),'passes':bool(trans<policy.trans_threshold and rot<policy.rot_threshold_deg)}


def _evaluate(entries,hosts,host_indices,*,batch_size=8,backend='cpu',policy=Protocol(),trace=False,slide_checkpoint_path=None,capacities=None,factory=None,trace_candidates=None,on_worlds=None):
    if batch_size<1 or len(entries)!=len(host_indices):raise ValueError('invalid batch mapping')
    ids=[e['id'] for e in entries]
    if len(set(ids))!=len(ids):raise ValueError('duplicate candidate ID')
    if any(i<0 or i>=len(hosts) for i in host_indices):raise ValueError('host index out of range')
    if not entries:
        return {'schema':'dexonomy-tabletop-slide-fc/v1','backend':backend,'slide':[],'directions':[],'combined':[],'snapshots':{},'traces':{},'counts':{'input':0,'slide_pass':0,'direction_tasks':0,'joint_pass':0},'protocol':vars(policy)}
    if backend not in ('cpu','warp'):raise ValueError('unknown backend')
    if factory is not None and backend!='warp':raise ValueError('external factory requires warp backend')
    if backend=='cpu':factory=CpuBatch
    elif factory is None:
        from dexonomy.tabletop_warp import WarpFactory
        factory=WarpFactory(hosts,capacities=capacities)
    selected=set(ids if trace and trace_candidates is None else (trace_candidates or [])) if trace else set()
    if not selected.issubset(ids):raise ValueError('unknown trace candidate')
    plans=[slide_plan(hosts[i],e['data']) for i,e in zip(host_indices,entries)]
    # Group by control length, never pad or step a completed Slide world.
    groups={}
    for i,p in enumerate(plans):groups.setdefault(len(p['segments']),[]).append(i)
    slides={};snapshots={};traces={};directions=[]
    source_root=Path(__file__).resolve().parent
    source_identity={str(p.relative_to(source_root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_root.rglob('*.py')}
    checkpoint_identity={'schema':'tabletop-slide-checkpoint/v1','ids':ids,'backend':backend,'policy':vars(policy),
                         'models':[hosts[i].tabletop_identity for i in host_indices],'source':source_identity,
                         'input_sha256':digest(entries),'trace':trace,'trace_candidates':sorted(selected)}
    checkpoint_path=Path(slide_checkpoint_path) if slide_checkpoint_path else None
    phase_started=time.perf_counter()
    resumed=False
    if checkpoint_path is not None and checkpoint_path.exists():
        saved=np.load(checkpoint_path,allow_pickle=True).item()
        if saved['identity']!=checkpoint_identity or saved['sha256']!=digest({k:v for k,v in saved.items() if k!='sha256'}):
            raise ValueError('Slide checkpoint source/input/model/policy/content mismatch')
        slides,snapshots,traces=saved['slides'],saved['snapshots'],saved['traces']
        if set(slides)!=set(ids) or set(snapshots)!={name for name in ids if slides[name]['passes']}:raise ValueError('incomplete Slide checkpoint')
        for i,name in enumerate(ids):
            if name in snapshots:check_snapshot(snapshots[name],hosts[host_indices[i]],name,'mujoco-3.17' if backend=='cpu' else 'mujoco-warp-3.17')
        resumed=True
    for group in ([] if resumed else groups.values()):
        for off in range(0,len(group),batch_size):
            indexes=group[off:off+batch_size];pp=[plans[i] for i in indexes]
            if on_worlds:on_worlds([(entries[i]['id'],-1) for i in indexes])
            batch=factory(hosts,[host_indices[i] for i in indexes],starts=[p['start'] for p in pp])
            capture_trace=any(ids[i] in selected for i in indexes)
            frames=[batch.qpos()] if capture_trace else []
            for seg in range(len(pp[0]['segments'])):
                batch.force([p['forces'][seg] for p in pp])
                for sub in range(5):
                    batch.control([p['segments'][seg][sub] for p in pp]);batch.step(5,contact=True)
                    if capture_trace:frames.append(batch.qpos())
            batch.step(50,contact=True);poses=batch.poses();states=batch.states()
            slide_flags,slide_near,slide_min=batch.contacts(policy.table_contact_tolerance_m)
            contacts=batch.end_contacts(policy.table_contact_tolerance_m)
            if capture_trace:frames.append(batch.qpos())
            for j,i in enumerate(indexes):
                name=ids[i];m=metrics(pp[j]['target'],poses[j],policy)
                slides[name]={'candidate_id':name,**m,'target':pp[j]['target'],'final_pose':poses[j],
                              'trajectory_table_contact_counts':slide_flags[j],'trajectory_table_margin_counts':slide_near[j],
                              'terminal_table_contacts':contacts[j],'reason':None if m['passes'] else 'slide_pose_error'}
                if name in selected:traces[name]={'slide_qpos':np.stack([f[j] for f in frames])}
                if m['passes']:
                    saved=snapshot(hosts[host_indices[i]].tabletop_identity,name,batch.backend,states[j],pp[j]['hold_target'],contacts[j])
                    snapshots[name]=saved
            del batch
    if checkpoint_path is not None and not resumed:
        checkpoint_path.parent.mkdir(parents=True,exist_ok=True)
        saved={'identity':checkpoint_identity,'slides':slides,'snapshots':snapshots,'traces':traces}
        saved['sha256']=digest(saved)
        temporary=checkpoint_path.with_suffix('.partial')
        with temporary.open('wb') as f:
            np.save(f,saved,allow_pickle=True)
            f.flush()
            import os
            os.fsync(f.fileno())
        temporary.replace(checkpoint_path)
    slide_wall_s=time.perf_counter()-phase_started
    phase_started=time.perf_counter()
    work=direction_tasks([i for i,name in enumerate(ids) if name in snapshots])
    for off in range(0,len(work),batch_size):
        chunk=work[off:off+batch_size];ss=[snapshots[ids[i]] for i,d in chunk]
        if on_worlds:on_worlds([(ids[i],d) for i,d in chunk])
        batch=factory(hosts,[host_indices[i] for i,d in chunk],snapshots=ss,candidate_ids=[ids[i] for i,d in chunk])
        start_states=batch.states()
        for saved,state in zip(ss,start_states):
            for k in STATE:
                expected=np.zeros_like(saved['fields'][k]) if k in ('qfrc_applied','xfrc_applied') else saved['fields'][k]
                if not np.array_equal(state[k],expected):raise ValueError('FC start restore differs: '+k)
        batch.force([DIRECTIONS[d] for i,d in chunk]);holds=[]
        capture_trace=any(ids[i] in selected for i,d in chunk)
        frames=[batch.qpos()] if capture_trace else []
        for _ in range(policy.holds):
            batch.step(policy.steps_per_hold,contact=True,tolerance=policy.table_contact_tolerance_m);holds.append(batch.poses())
            if capture_trace:frames.append(batch.qpos())
        flags,near,minimum=batch.contacts(policy.table_contact_tolerance_m)
        for j,(i,d) in enumerate(chunk):
            name=ids[i];checks=[metrics(slides[name]['final_pose'],pose[j],policy) for pose in holds]
            touched=bool(np.any(flags[j]) or np.any(ss[j]['initial_table_contacts']))
            ok=all(x['passes'] for x in checks) and not touched
            directions.append({'candidate_id':name,'direction_index':d,'direction':DIRECTIONS[d],
                'passes':ok,'pose_checks':checks,'table_contact_counts':flags[j],'margin_only_counts':near[j],
                'minimum_table_distance_m':[float(x) if np.isfinite(x) else None for x in minimum[j]],
                'reason':'table_recontact' if touched else None if ok else 'fc_pose_error',
                'start_snapshot_sha256':ss[j]['sha256'],'batch_index':off//batch_size})
            if name in selected:traces[name][f'fc_{d}_qpos']=np.stack([f[j] for f in frames])
        del batch
    combined=aggregate(ids,slides,directions)
    return {'schema':'dexonomy-tabletop-slide-fc/v1','backend':backend,'slide_checkpoint_reused':resumed,'timing':{'slide_s':slide_wall_s,'fc_s':time.perf_counter()-phase_started},'slide':list(slides.values()),'directions':directions,
            'combined':combined,'snapshots':snapshots,'traces':traces,'counts':{'input':len(ids),'slide_pass':len(snapshots),
            'direction_tasks':len(directions),'joint_pass':sum(x['passes'] for x in combined)},'protocol':vars(policy)}


def evaluate(entries,hosts,host_indices,*,batch_size=8,backend='cpu',policy=Protocol(),trace=False,
             slide_checkpoint_path=None,capacities=None,factory=None,trace_candidates=None,isolation_cfg=None):
    kwargs=dict(batch_size=batch_size,backend=backend,policy=policy,trace=trace,
                slide_checkpoint_path=slide_checkpoint_path,capacities=capacities,
                factory=factory,trace_candidates=trace_candidates)
    if backend!='warp' or not entries:
        return _evaluate(entries,hosts,host_indices,**kwargs)
    from types import SimpleNamespace
    from dexonomy.op.simulation_isolation import execute_isolated,known_error
    from dexonomy.sim.mujoco_warp_env import MuJoCoWarpAllocationError,release_warp_scene_memory
    from dexonomy.tabletop_warp import WarpFactory
    factory=factory or WarpFactory(hosts,capacities=capacities)
    kwargs['factory']=factory
    cfg=isolation_cfg or {}
    candidates={e['id']:SimpleNamespace(source_name=e['id'],data=e['data']) for e in entries}
    prior=set();cached=[]
    if cfg.get('isolate_eval_epa',False):
        for candidate in candidates.values():
            row=known_error(cfg.simulation_error_root,candidate)
            if row:cached.append(row);prior.add(candidate.source_name)
        if len(prior)>int(cfg.get('simulation_error_limit',1)):raise RuntimeError('Simulation error ledger limit exceeded')
    worlds=[(c,-1) for c in candidates.values() if c.source_name not in prior]
    current=[]
    def mapping(rows):
        current[:]=[(candidates[name],direction) for name,direction in rows]
    def execute(active):
        names={c.source_name for c,d in active};indexes=[i for i,e in enumerate(entries) if e['id'] in names]
        options=dict(kwargs,on_worlds=mapping)
        if trace_candidates is not None:options['trace_candidates']=[name for name in trace_candidates if name in names]
        # Original checkpoint remains immutable evidence after an isolation.
        if len(indexes)!=len(entries) and slide_checkpoint_path is not None:
            path=Path(slide_checkpoint_path)
            options['slide_checkpoint_path']=path.with_name(path.stem+'-healthy-'+digest(sorted(names))+path.suffix)
        try:return _evaluate([entries[i] for i in indexes],hosts,[host_indices[i] for i in indexes],**options)
        except MuJoCoWarpAllocationError as error:
            error.worlds=list(current)
            raise
    def reset():
        factory.release_execution()
        release_warp_scene_memory('cuda:0')
    if worlds:
        _,report,isolated,attempts=execute_isolated(worlds,execute,reset,cfg,prior)
    else:report=None;isolated=[];attempts=[]
    errors=cached+isolated
    if report is None:report=_evaluate([],hosts,[],backend=backend)
    bad={r['source_name'] for r in errors}
    report['simulation_errors']=errors;report['recovery_attempts']=attempts
    report['slide'].extend(dict(candidate_id=name,passes=False,reason='simulation_error') for name in bad)
    report['combined'].extend(dict(candidate_id=name,passes=False) for name in bad)
    order={e['id']:i for i,e in enumerate(entries)}
    for key in ('slide','combined'):report[key].sort(key=lambda r:order[r['candidate_id']])
    report['counts']['input']=len(entries);report['counts']['simulation_error']=len(errors)
    report['runtime']=factory.stats
    return report


def aggregate(ids,slides,directions):
    rows={name:{} for name in ids}
    if set(slides)!=set(ids):raise ValueError('Slide inventory incomplete')
    for row in directions:
        name=row['candidate_id'];d=row['direction_index']
        if name not in rows:raise ValueError('unknown candidate')
        add_direction(rows[name],d,row)
    result=[]
    for name in ids:
        expected=set(range(6)) if slides[name]['passes'] else set()
        if set(rows[name])!=expected:raise ValueError('incomplete or unexpected direction set')
        result.append({'candidate_id':name,'passes':bool(slides[name]['passes'] and all(r['passes'] for r in ordered_directions(rows[name])))})
    return result
