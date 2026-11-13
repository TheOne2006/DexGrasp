"""Explicit bounded candidate isolation; unknown/global errors remain fatal."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def implicated_candidates(worlds, diagnostics):
    flags=diagnostics.get('allocation_overflow_flags',[])
    if len(flags)!=len(worlds) or not any(flags):raise ValueError('unlocalized simulation error')
    if any(int(flag) not in (0,256) for flag in flags):raise ValueError('unsupported simulation error')
    for value,capacity in [('nacon','nacon_capacity'),('ncollision','ccd_capacity'),('nefc','nefc_capacity')]:
        if diagnostics.get(value,0)>diagnostics.get(capacity,0):raise ValueError('global allocation overflow')
    return {worlds[i][0].source_name for i,f in enumerate(flags) if f}


def discard_batch_state(model_bank, device):
    from dexonomy.sim.data_workspace import clear
    from dexonomy.sim.mujoco_warp_env import release_warp_scene_memory
    if model_bank is not None and hasattr(model_bank,'_eval_workspace'):model_bank._eval_workspace.release()
    clear()
    release_warp_scene_memory(device)


def record_error(root,candidate,directions,diagnostics,*,reason="EPA_HORIZON",stage="eval"):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    path=root/(hashlib.sha256(candidate.source_name.encode()).hexdigest()+'.npy')
    from dexonomy.acceptance_audit import serial
    input_hash=hashlib.sha256(json.dumps(candidate.data,sort_keys=True,default=serial).encode()).hexdigest()
    scene=Path(candidate.data.get('scene_path','unknown')).parent.name
    metadata=getattr(candidate,'error_scene_metadata',{})
    record=dict(candidate_id=candidate.source_name,scene_id=metadata.get('scene_id',scene),object_id=metadata.get('object_id',scene.rsplit('_scale',1)[0]),scale=np.asarray(candidate.data.get('obj_scale',[])).tolist(),family=candidate.data.get('tmpl_name'),input=candidate.data,input_hash=input_hash,directions=directions,invalidated_directions=list(range(6)),error=reason,stage=stage,diagnostics=diagnostics,status='simulation_error',success=False)
    if path.exists():
        old=np.load(path,allow_pickle=True).item()
        if old['candidate_id']!=candidate.source_name or old['input_hash']!=input_hash or old.get('error','EPA_HORIZON')!=reason or old.get('stage','eval')!=stage:raise RuntimeError('isolation ledger identity mismatch')
    else:
        temp=path.with_suffix('.partial')
        with temp.open('wb') as f:np.save(f,record,allow_pickle=True);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    return dict(source_name=candidate.source_name,success=False,output_data=None,output_kind=None,metrics=dict(type='simulation_error',error=reason,stage=stage,directions=directions,ledger=str(path)))


def known_error(root,candidate):
    path=Path(root)/(hashlib.sha256(candidate.source_name.encode()).hexdigest()+'.npy')
    if not path.exists():return None
    saved=np.load(path,allow_pickle=True).item()
    return record_error(root,candidate,saved['directions'],saved['diagnostics'],reason=saved.get('error','EPA_HORIZON'),stage=saved.get('stage','eval'))


def execute_isolated(worlds, execute, reset, cfg, prior_ids=None):
    """Shared bounded recovery: discard poisoned batch and replay healthy inputs.

    A protocol may attach the exact active world mapping to an allocation error.
    Only localized EPA_HORIZON is recoverable; OOM/unknown errors propagate.
    Removed candidates lose every direction, including any prior partial result.
    """
    import time
    from dexonomy.sim.mujoco_warp_env import MuJoCoWarpAllocationError
    remaining=list(worlds);isolated=[];attempts=[]
    prior_ids=set() if prior_ids is None else prior_ids
    for attempt in range(4):
        started=time.perf_counter()
        try:
            report=execute(remaining)
        except MuJoCoWarpAllocationError as error:
            failure=(str(error),dict(error.diagnostics))
            mapping=list(getattr(error,'worlds',remaining))
        except BaseException:
            reset()
            raise
        else:
            attempts.append(dict(duration_s=time.perf_counter()-started,status='PASS',worlds=len(remaining)))
            return remaining,report,isolated,attempts
        reset()
        attempts.append(dict(duration_s=time.perf_counter()-started,status='FAIL',worlds=len(remaining),diagnostics=failure[1]))
        if not cfg.get('isolate_eval_epa',False):raise MuJoCoWarpAllocationError(*failure)
        try:bad=implicated_candidates(mapping,failure[1])
        except ValueError:raise MuJoCoWarpAllocationError(*failure) from None
        if attempt>=3 or len(prior_ids|bad)>int(cfg.get('simulation_error_limit',1)):
            raise MuJoCoWarpAllocationError('Simulation error rate limit exceeded',failure[1])
        for name in sorted(bad):
            candidate=next(c for c,d in mapping if c.source_name==name)
            directions=[d for i,(c,d) in enumerate(mapping) if c.source_name==name and failure[1]['allocation_overflow_flags'][i]]
            isolated.append(record_error(cfg.simulation_error_root,candidate,directions,failure[1]))
            for attr in ('checkpoint_poses','solver_flags'):
                if hasattr(candidate,attr):getattr(candidate,attr).clear()
        prior_ids.update(bad)
        remaining=[(c,d) for c,d in remaining if c.source_name not in bad]
        if not remaining:return [],None,isolated,attempts
    raise AssertionError('unreachable isolation attempt')


class RefinementNonfiniteError(RuntimeError):
    """Typed, world-indexed error; never interpreted from a free-text message."""
    def __init__(self, diagnostics):
        self.diagnostics=diagnostics
        super().__init__('resident refinement non-finite state: '+str(diagnostics))


def nonfinite_candidates(names, indices, diagnostics):
    flags=diagnostics.get('history_flags',[])
    if len(flags)!=len(indices) or not any(flags):raise ValueError('unlocalized nonfinite error')
    if any(type(f) is not int or f not in (0,536870912) for f in flags):raise ValueError('mixed or unsupported nonfinite flags')
    if any(type(i) is not int or i<0 or i>=len(names) for i in indices):raise ValueError('invalid physical world mapping')
    return {names[indices[i]] for i,f in enumerate(flags) if f}


def execute_refine_isolated(data,names,execute,reset,cfg,prior_ids):
    """Replay whole healthy logical batch once from immutable inputs after reset.

    execute returns results plus physical-to-logical mapping on typed failures.
    A second failure in the rebuilt batch is systemic and stops, even if localized.
    """
    from types import SimpleNamespace
    pending=list(range(len(data)));saved={};attempts=[]
    if len(set(names))!=len(names):raise ValueError('duplicate logical candidate IDs')
    enabled=bool(cfg.get('isolate_refine_nonfinite',False))
    limit=int(cfg.get('simulation_error_limit',1))
    if enabled:
        for i,(name,item) in enumerate(zip(names,data)):
            found=known_error(cfg.refine_simulation_error_root,SimpleNamespace(source_name=name,data=item))
            if found:
                if found['metrics']['error']!='nonfinite' or found['metrics']['stage']!='refine':raise RuntimeError('Refine ledger reason mismatch')
                saved[i]=found;prior_ids.add(name)
        if len(prior_ids)>limit:raise RuntimeError('Refine simulation error limit exceeded')
        pending=[i for i in pending if i not in saved]
    for attempt in range(2):
        if not pending:break
        try:
            result=execute([data[i] for i in pending],[names[i] for i in pending])
        except RefinementNonfiniteError as error:
            failure=dict(error.diagnostics);mapping=list(getattr(error,'world_indices',[]))
        except BaseException:
            reset();raise
        else:
            if len(result)!=len(pending):raise RuntimeError('Refine result mapping mismatch')
            saved.update(zip(pending,result));break
        # Outside except: traceback must release all poisoned Data/Graphs first.
        reset()
        if not enabled:raise RefinementNonfiniteError(failure)
        try:bad=nonfinite_candidates([names[i] for i in pending],mapping,failure)
        except ValueError:raise RefinementNonfiniteError(failure) from None
        if attempt or len(prior_ids|bad)>limit:raise RefinementNonfiniteError(dict(failure,stop='nonfinite rate or rebuilt batch failure'))
        for i in pending:
            if names[i] in bad:
                if not np.isfinite(np.asarray(data[i]['grasp_qpos'])).all():raise RuntimeError('Nonfinite original input; no isolation')
                saved[i]=record_error(cfg.refine_simulation_error_root,SimpleNamespace(source_name=names[i],data=data[i]),[],failure,reason='nonfinite',stage='refine')
        prior_ids.update(bad)
        pending=[i for i in pending if names[i] not in bad]
    if len(saved)!=len(data):raise RuntimeError('Refine isolation lost candidates')
    return [saved[i] for i in range(len(data))]


class EvaluationNonfiniteError(RuntimeError):
    """Exact flags in the physical-world order supplied to the Eval simulator."""
    def __init__(self, diagnostics):
        self.diagnostics=diagnostics
        super().__init__('Eval non-finite state: '+str(diagnostics))


def require_finite_input(value):
    if isinstance(value,dict):
        for v in value.values():require_finite_input(v)
    elif isinstance(value,(tuple,list)):
        for v in value:require_finite_input(v)
    elif isinstance(value,(np.ndarray,np.number,float,complex)):
        a=np.asarray(value)
        if a.dtype.kind in 'fc' and not np.isfinite(a).all():
            raise RuntimeError('Nonfinite original input; no isolation')


def execute_eval_isolated(worlds, execute, reset, cfg, prior_ids):
    """Discard failed batch, invalidate every bad candidate direction, replay once.

    Worlds carry the actual candidate and direction, including duplicated slots.
    Existing verified directions of healthy candidates are independent of this
    batch. No returned result from a failed execute call is consumed.
    """
    import time
    from dexonomy.sim.mujoco_warp_env import MuJoCoWarpAllocationError
    remaining=list(worlds);isolated=[];attempts=[];rebuilt_nonfinite=False
    for attempt in range(4):
        started=time.perf_counter()
        try:report=execute(remaining)
        except EvaluationNonfiniteError as error:
            failure=dict(error.diagnostics);kind='nonfinite'
        except MuJoCoWarpAllocationError as error:
            failure=dict(error.diagnostics);kind='EPA_HORIZON';message=str(error)
        except BaseException:
            reset();raise
        else:
            attempts.append(dict(duration_s=time.perf_counter()-started,status='PASS',worlds=len(remaining)))
            return remaining,report,isolated,attempts
        # The caught traceback no longer owns poisoned simulator arrays here.
        reset()
        attempts.append(dict(duration_s=time.perf_counter()-started,status='FAIL',worlds=len(remaining),error=kind,diagnostics=failure))
        if rebuilt_nonfinite:
            raise EvaluationNonfiniteError(dict(failure,stop='rebuilt batch failed'))
        if kind=='nonfinite':
            if not cfg.get('isolate_eval_nonfinite',False):raise EvaluationNonfiniteError(failure)
            try:
                names=[c.source_name for c,d in remaining]
                if any(type(d) is not int or not 0<=d<6 for c,d in remaining):raise ValueError('direction mapping')
                bad=nonfinite_candidates(names,list(range(len(names))),failure)
            except ValueError:raise EvaluationNonfiniteError(failure) from None
            flags=failure['history_flags'];rebuilt_nonfinite=True
        else:
            if not cfg.get('isolate_eval_epa',False):raise MuJoCoWarpAllocationError(message,failure)
            try:bad=implicated_candidates(remaining,failure)
            except ValueError:raise MuJoCoWarpAllocationError(message,failure) from None
            flags=failure['allocation_overflow_flags']
        if attempt>=3 or len(prior_ids|bad)>int(cfg.get('simulation_error_limit',1)):
            raise RuntimeError('Simulation error rate limit exceeded: '+str(failure))
        for name in sorted(bad):
            candidate=next(c for c,d in remaining if c.source_name==name)
            require_finite_input(candidate.data)
            directions=sorted({d for i,(c,d) in enumerate(remaining) if c.source_name==name and flags[i]})
            isolated.append(record_error(cfg.simulation_error_root,candidate,directions,failure,reason=kind,stage='eval'))
            for attr in ('checkpoint_poses','checkpoint_seen','terminal_state','solver_flags','batch_ids'):
                if hasattr(candidate,attr):getattr(candidate,attr).clear()
        prior_ids.update(bad)
        remaining=[(c,d) for c,d in remaining if c.source_name not in bad]
        if not remaining:return [],None,isolated,attempts
    raise AssertionError('unreachable isolation attempt')


def nonfinite_stop_reason(rows,policy):
    """Recent committed-job denominator; persistent rows survive migration."""
    count=lambda r:int(r.get('nonfinite',0))
    consecutive=0
    for row in reversed(rows):
        if not count(row):break
        consecutive+=1
    if consecutive>=int(policy.get('nonfinite_consecutive_jobs',2)):
        return 'consecutive nonfinite jobs'
    recent=rows[-int(policy.get('nonfinite_window_jobs',20)):]
    errors=sum(count(r) for r in recent);denominator=sum(r.get('refined',0) for r in recent)
    if len(recent)>=2 and errors>=int(policy.get('nonfinite_window_min_errors',4)) and errors/max(1,denominator)>=float(policy.get('nonfinite_window_fraction',.01)):
        return 'recent nonfinite candidate fraction'
    return None
