"""Bounded execution capacities; physics and candidate identities are unchanged."""
import re

def key(config,group,family,stage):
    return f"rank{config['rank']:03d}/g{group:03d}/{family}/{stage}"

def ladder(config,group,stage):
    base=int(config['groups'][group].get(stage+'_batch',config[stage+'_batch']))
    policy=config.get('capacity_policy',{})
    values=policy.get(stage+'_ladder',[base])
    return sorted({int(x) for x in values if 0<int(x)<=base}|{base},reverse=True)

def selected(config,state,group,family,stage):
    k=key(config,group,family,stage);levels=ladder(config,group,stage)
    choice=state.get('capacity_choices',{}).get(k,{})
    policy=config.get('capacity_policy',{})
    # An explicitly reviewed bank fallback applies only to unseen combinations.
    # Existing per-family choices (including healthy larger batches) take priority.
    bank=f"rank{config['rank']:03d}/g{group:03d}/{stage}"
    fallback=policy.get('bank_initial_choices',{}).get(bank,levels[0])
    n=int(choice.get('batch',policy.get('initial_choices',{}).get(k,fallback)))
    if adaptive(config) and not choice.get('validated'):
        learned=learned_start(config,state,group,stage)
        if learned is not None:n=min(n,learned)
    if n not in levels:raise ValueError('Capacity not in authorized ladder: '+k)
    return n

def successful(config,state,group,family,stage,count):
    if not config.get('capacity_policy') or not count:return
    k=key(config,group,family,stage);n=selected(config,state,group,family,stage)
    previous=state.setdefault('capacity_choices',{}).get(k,{})
    if previous.get('validated') and previous.get('batch')==n:return
    state['capacity_choices'][k]={'batch':n,'validated':True}
    state.setdefault('capacity_history',[]).append(dict(key=k,batch=n,event='validated',candidates=count))

def lower(config,state,group,family,stage,*,attempt,reason,input_sha256):
    policy=config['capacity_policy'];k=key(config,group,family,stage);history=state.setdefault('capacity_history',[])
    failures=[x for x in history if x['event']=='capacity_failure']
    if len(failures)>=policy['max_failures_per_rank']:raise RuntimeError('Rank capacity failure limit reached')
    if sum(x['key']==k for x in failures)>=policy['max_failures_per_key']:raise RuntimeError('Capacity key failure limit reached')
    old=selected(config,state,group,family,stage);levels=ladder(config,group,stage);index=levels.index(old)
    if index+1==len(levels):raise RuntimeError('Minimum authorized capacity failed')
    if any(x['attempt']==attempt for x in failures):raise RuntimeError('Attempt already handled')
    new=levels[index+1];record=dict(event='capacity_failure',key=k,batch=old,next_batch=new,attempt=attempt,reason=reason,input_sha256=input_sha256)
    history.append(record);state.setdefault('capacity_choices',{})[k]={'batch':new,'validated':False};return record

def capacity_failure(result,resources,console):
    if result.get('status')=='PASS':return False
    if any(r.get('unexpected_gpu_pids') or r.get('ownership_live') is False for r in resources):return False
    if any(s in console for s in ['non-finite','simulation is unstable','allocation overflow','identity mismatch','Checkpoint changed','nonfinite','NaN','Inf detected','checksum','corrupt','cgroup_memory','host_memory']):return False
    reason=result.get('stop_reason');oom=bool(re.search(r'CUDA_ERROR_OUT_OF_MEMORY|CUDA error 2\b|OutOfMemoryError: CUDA out of memory',console))
    return reason=='gpu_headroom_less_than_2GiB' or (reason in (None,'runtime_safety_error') and oom)

# Pure, serializable policy surface shared by stage-specific drivers. No CUDA,
# file I/O, simulator imports or physics defaults belong in this module.
def adaptive(config):
    return config.get('capacity_policy',{}).get('mode')=='adaptive/v1'

def revision(config):
    value=config['capacity_policy'].get('revision')
    if not isinstance(value,str) or not value:raise ValueError('Explicit capacity policy revision required')
    return value

def initialize(config,state):
    if not adaptive(config):return
    revision(config)
    policy=config['capacity_policy']
    for name in ('window_jobs','window_min_jobs','consecutive_deferred','no_progress_failures','no_progress_jobs','bank_window','bank_min_families'):
        if not isinstance(policy.get(name),int) or policy[name]<1:raise ValueError('Invalid capacity bound: '+name)
    if not 0<policy.get('deferred_fraction',0)<=1:raise ValueError('Invalid deferred fraction')
    if policy['window_min_jobs']>policy['window_jobs']:raise ValueError('Invalid capacity window')
    if 'capacity_events' not in state:
        state['capacity_events']=[]
        state['capacity_adaptive_boundary']={'revision':revision(config),'legacy_history_entries':len(state.get('capacity_history',[]))}
    state.setdefault('capacity_deferred',{})

def bank_key(config,group,stage):
    return f"rank{config['rank']:03d}/g{group:03d}/{stage}"

def learned_start(config,state,group,stage):
    policy=config['capacity_policy'];b=bank_key(config,group,stage)
    failures=[e for e in state.get('capacity_events',[]) if e['event']=='failure' and e['bank']==b][-policy['bank_window']:]
    targets={}
    for e in failures:
        target=e['next_batch'] if e['next_batch'] is not None else e['batch']
        targets[e['family']]=min(targets.get(e['family'],target),target)
    support=policy['bank_min_families']
    if len(targets)<support:return None
    # Every learned starting level needs support from distinct families. One
    # pathological family reaching the floor must not drag its entire bank down.
    return sorted(targets.values())[support-1]

def deferred_for(config,state,group,family):
    for stage in ('refine','eval'):
        record=state.get('capacity_deferred',{}).get(key(config,group,family,stage))
        if record and record.get('status')=='deferred':return record
    return None

def retry_allowed(config,record):
    return (record['revision']!=revision(config) and
            record['key'] in config['capacity_policy'].get('retry_deferred_keys',[]))

def blocked(config,state,group,family,tag):
    record=deferred_for(config,state,group,family)
    if record is None:return False
    # Only the frozen original input can be retried, never a new tour's sample.
    return not (retry_allowed(config,record) and record['tag']==tag)

def completed(config,state,tag,init_count):
    if not adaptive(config):return
    initialize(config,state)
    events=state['capacity_events']
    if any(e['event']=='completed' and e['tag']==tag for e in events):return
    events.append(dict(event='completed',tag=tag,init_count=init_count))
    for record in state['capacity_deferred'].values():
        if record['tag']==tag:record['status']='resolved'

def systemic_reason(config,state):
    if not adaptive(config):return None
    p=config['capacity_policy'];events=state.get('capacity_events',[])
    terminal=[e for e in events if e['event'] in ('completed','deferred')]
    # A deferred job retried under a new explicit policy is one recent outcome.
    seen=set();unique=[]
    for e in reversed(terminal):
        if e['tag'] not in seen:unique.append(e);seen.add(e['tag'])
    window=unique[:p['window_jobs']]
    consecutive=[]
    for e in unique:
        if e['event']!='deferred':break
        consecutive.append(e)
    if len(consecutive)>=p['consecutive_deferred']:return 'consecutive_minimum_capacity_failures'
    failures=[e for e in window if e['event']=='deferred']
    if len(window)>=p['window_min_jobs'] and len(failures)/len(window)>=p['deferred_fraction'] and len(failures)>=p['consecutive_deferred']:
        return 'recent_minimum_capacity_failure_fraction'
    recent=[]
    for e in reversed(events):
        if e['event']=='completed' and e['init_count']>0:break
        if e['event']=='failure':recent.append(e)
    if len(recent)>=p['no_progress_failures'] and len({e['tag'] for e in recent})>=p['no_progress_jobs']:
        return 'capacity_failures_without_committed_progress'
    return None

def adapt_failure(config,state,group,family,stage,*,tag,attempt,reason,input_sha256,input_path,candidate_ids):
    initialize(config,state);k=key(config,group,family,stage)
    if any(e.get('attempt')==attempt for e in state['capacity_events']):raise RuntimeError('Attempt already handled')
    old=selected(config,state,group,family,stage);levels=ladder(config,group,stage);index=levels.index(old)
    new=levels[index+1] if index+1<len(levels) else None
    record=dict(event='capacity_failure',key=k,tag=tag,batch=old,next_batch=new,attempt=attempt,reason=reason,input_sha256=input_sha256,revision=revision(config))
    state.setdefault('capacity_history',[]).append(record)
    state['capacity_events'].append(dict(record,event='failure',bank=bank_key(config,group,stage),family=family))
    if new is not None:
        state.setdefault('capacity_choices',{})[k]={'batch':new,'validated':False}
    else:
        deferred=dict(record,event='deferred',status='deferred',input_path=input_path,candidate_ids=list(candidate_ids),group=group,family=family,stage=stage)
        state['capacity_deferred'][k]=deferred
        state['capacity_events'].append(dict(event='deferred',tag=tag,key=k,attempt=attempt))
    return record

def all_combinations_deferred(config,state,families):
    if not adaptive(config):return False
    records=[deferred_for(config,state,g,f) for g in range(len(config['groups'])) for f in families]
    return bool(records) and all(r is not None and not retry_allowed(config,r) for r in records)
