"""Bounded strict quotas and stage transactions using the existing stage operators."""
import argparse, gc, hashlib, importlib.metadata, json, os, random, time
from pathlib import Path
from night_schedule import tour_numbers, planned_visits, reached
from night_config import validate_worker_mode, stage_configs
from night_transactions import checkpoint_artifacts, apply_feedback, completion_row, record_completion
from night_templates import install_snapshot
import numpy as np
from dexonomy.cli import check_assets
from dexonomy.acceptance_audit import event, serial


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=serial).encode()).hexdigest()


def atomic(path, value):
    write_started = time.perf_counter()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    with temp.open('wb') as f:
        np.save(f, value, allow_pickle=True); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)
    event('stage_artifact_write',path=str(path),wall_s=time.perf_counter()-write_started,bytes=path.stat().st_size)


def read(path):
    return np.load(path, allow_pickle=True).item()


def pending_scenes(counts,ledger,round_id,target,max_rounds):
    # An interrupted attempt has unknown cost but still consumes its scene budget.
    completed={key for row in ledger if row['round']==round_id and row['status']=='PASS' for key in row['scenes']}
    attempts={key:sum(key in row['scenes'] for row in ledger) for key in counts}
    return [key for key,n in counts.items() if n<target and key not in completed and attempts[key]<max_rounds]


class DrainComplete(Exception):pass

def main():
    p = argparse.ArgumentParser(); p.add_argument('config', type=Path); p.add_argument('output', type=Path)
    a = p.parse_args(); c = json.loads(a.config.read_text())
    mode = validate_worker_mode(c)
    # Runtime imports can initialize Warp/TorchScript: never execute on import or --help.
    import torch
    from dexonomy.upstream_stream import _compose_cfg, _common_overrides, _grasp_overrides
    from dexonomy.op.gen_init import operate_init
    from dexonomy.data.family_pool import FamilyPoolLoader
    from dexonomy.op.gen_grasp_warp_resident import MixedRefinementCache, refine_grasp_batch_mixed
    from dexonomy.op.eval_force_closure_warp_mixed import MixedForceClosureBatcher
    from dexonomy.sim.mujoco_warp_env import release_warp_scene_memory
    from dexonomy.sim import MuJoCo_VisEnv, HandCfg
    from dexonomy.op.simulation_isolation import nonfinite_stop_reason
    tabletop = mode == 'tabletop'
    batcher_type = MixedForceClosureBatcher
    if tabletop:
        from dexonomy.tabletop_batcher import TabletopSlideFCBatcher
        batcher_type = TabletopSlideFCBatcher
    def stop_init():
        return reached(c.get('stop_new_init_epoch'),time.time()) or Path(os.environ.get('DEXONOMY_STOP_REQUEST',str(a.output/'stop-request.json'))).exists()
    out = Path(c.get('checkpoint_root', a.output)); out.mkdir(parents=True, exist_ok=True)
    from night_checkpoint_identity import token as checkpoint_token
    token = checkpoint_token(c,Path(__file__))
    dataset_id=c.get('dataset_id',out.name)
    assert dataset_id and '/' not in dataset_id and dataset_id not in ('.','..')
    statefile = out/'state.npy'
    state = read(statefile) if statefile.exists() else {'schema':c['checkpoint_schema'],'config_hash':token,'completed':[], 'rows':[], 'family_archive':{}}
    assert state.get('schema')==c['checkpoint_schema'], 'Legacy checkpoint migration is not implicit'
    assert state['config_hash'] == token, 'Resume source/config changed'
    for rel,expected in state.get('artifacts',{}).items():
        assert hashlib.sha256((out/rel).read_bytes()).hexdigest()==expected, 'Checkpoint changed: '+rel
    assert min(c[k] for k in ['target','sample_points','object_batch','refine_batch','eval_batch','epochs_per_visit'])>0
    assert c.get('max_rounds',3)>0
    versions={n:importlib.metadata.version(n) for n in ['torch','mujoco','mujoco-warp','warp-lang','numpy','qpsolvers','clarabel']}
    assert versions['mujoco']=='3.17.0' and versions['mujoco-warp']=='3.17.0' and versions['warp-lang']=='1.16.0'
    environment=out/'environment.json'
    if environment.exists():assert json.loads(environment.read_text())==versions
    else:environment.write_text(json.dumps(versions,indent=2))
    check_assets(c['asset_root'])
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
    from omegaconf import open_dict
    refine_cfg, eval_cfg, base_init, seed_cfg = stage_configs(
        c, out, tabletop, _compose_cfg, _common_overrides, _grasp_overrides, open_dict)
    families = c.get('families') or sorted(f.stem for f in Path(seed_cfg.init_tmpl_dir).glob('*.npy'))
    visits=planned_visits(c)
    assert visits is None or visits>0
    state['planned_jobs']=None if visits is None else visits*c['epochs_per_visit']*c.get('families_per_visit',len(families))
    state['dataset_id']=dataset_id
    from night_capacity_policy import initialize, blocked, completed as capacity_completed, systemic_reason, all_combinations_deferred
    initialize(c, state)
    if systemic_reason(c, state):raise RuntimeError('Systemic capacity protection: '+systemic_reason(c, state))
    install_snapshot(c,state,out)
    atomic(statefile,state)
    loader = FamilyPoolLoader(xml_path=seed_cfg.hand.xml_path,skt_path=seed_cfg.hand_skt_path,
        init_tmpl_dir=seed_cfg.init_tmpl_dir,new_tmpl_dir=None,tmpl_names=families,
        selection_seed=c['seed'],family_capacity=c.get('family_capacity',512),
        n_worker=c.get('loader_workers',1),feedback_max_evo=seed_cfg.tmpl_max_evo)
    holder = {'loader':loader}
    if 'template_buffer' in state:
        loader.data_buffer = state['template_buffer']; loader.feedback_stats = state['template_stats']
    loader.feedback_seen_ids=set(state.get('feedback_seen_ids',[]))
    state['feedback_mode']='synchronous official prepare/append, committed with job completion'
    loader.feedback_stats['enabled']=True
    event('night_pool_policy',family_capacity=loader.family_capacity,distribution=loader.distribution(),memory=loader.memory_estimate())
    started = time.perf_counter(); caches = {}; e=None; draining=False
    def release_transient():
        from dexonomy.sim.data_workspace import clear
        for _,eval_cache in caches.values():
            eval_cache.release_execution()
        clear();gc.collect();release_warp_scene_memory('cuda:0');torch.cuda.empty_cache()
    def release_stage_transient(stage):
        before={'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()}
        release_transient()
        event('night_stage_memory_release',stage=stage,before=before,after={'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()})
    def checkpoint(stage):
        checkpoint_artifacts(state, out, stage)
        atomic(statefile,state)
        event('vnext_checkpoint',stage=stage,completed=len(state['completed']))
        if c.get('stop_after_stage') == stage:
            (a.output/'intentional-stop.json').parent.mkdir(parents=True,exist_ok=True)
            (a.output/'intentional-stop.json').write_text(json.dumps({'stage':stage,'checkpoint':str(statefile)}))
            raise SystemExit(75)
    try:
        for tour in tour_numbers(c,state.get('next_tour',0)):
            if all_combinations_deferred(c,state,families):raise RuntimeError('All scheduled combinations capacity deferred; no work remains until explicit policy change')
            for group, g in enumerate(c['groups']):
                if visits is not None and tour*len(c['groups'])+group>=visits:break
                refine_batch=int(g.get('refine_batch',c['refine_batch']))
                eval_batch=int(g.get('eval_batch',c['eval_batch']))
                with open_dict(refine_cfg):
                    refine_cfg.warp_world_batch_size=refine_batch
                    refine_cfg.refine_njmax=g.get('refine_njmax',c.get('refine_njmax',0))
                with open_dict(eval_cfg):eval_cfg.eval_njmax=g.get('eval_njmax',c.get('eval_njmax',0))
                with open_dict(refine_cfg):
                    for key in ['refine_nconmax','refine_nccdmax']:refine_cfg[key]=g.get(key,c.get(key,0))
                with open_dict(eval_cfg):
                    for key in ['eval_nconmax','eval_nccdmax']:eval_cfg[key]=g.get(key,c.get(key,0))
                cases = json.loads(Path(g['index']).read_text())['cases']
                all_by_scene = {x['scene_id']:x for x in cases}
                assert len(all_by_scene)==len(cases)
                visit= tour*len(c['groups'])+group
                count=c.get('families_per_visit',len(families))
                assert 1<=count<=len(families)
                active_families=[families[(visit*count+i)%len(families)] for i in range(count)]
                for epoch in range(c['epochs_per_visit']):
                    for family_index,family in enumerate(active_families):
                        active_cases=cases[family_index::count] if c.get('partition_scenes_by_family',False) else cases
                        by_scene={x['scene_id']:x for x in active_cases}
                        tag = f't{tour:02d}-g{group:03d}-e{epoch:02d}-{family}'
                        if tag in state['completed']:continue
                        if blocked(c,state,group,family,tag):
                            event('night_capacity_deferred_skip',tag=tag,group=group,family=family)
                            continue
                        job = out/'jobs'/tag
                        if stop_init() and not ((job/'init.npy').exists() or (job/'init-progress.npy').exists()):
                            draining=True;event('night_drained',completed=len(state['completed']),next_tag=tag);raise DrainComplete
                        job.mkdir(parents=True,exist_ok=True)
                        from night_capacity_policy import selected, successful
                        refine_batch=selected(c,state,group,family,'refine')
                        eval_batch=selected(c,state,group,family,'eval')
                        with open_dict(refine_cfg):
                            refine_cfg.warp_world_batch_size=refine_batch
                            refine_cfg.capacity_memory_audit=c.get('capacity_memory_audit',False)
                        if group in caches:
                            assert not caches[group][1].pending
                            caches[group][1].world_batch_size=eval_batch
                        event('night_capacity_selected',tag=tag,group=group,family=family,refine_batch=refine_batch,eval_batch=eval_batch)
                        activated=loader.activate_family(family,state['family_archive'].get(family,[]),out,c['seed']+visit*1000+epoch)
                        job_seed=int(digest([c['seed'],tag])[:8],16)
                        random.seed(job_seed);np.random.seed(job_seed);torch.manual_seed(job_seed)
                        event('night_family_activated',sampling_seed=job_seed,tag=tag,family=family,active_ids=activated,distribution=loader.distribution())
                        release_transient()
                        event('vnext_job_start',tag=tag,group=group,tour=tour,epoch=epoch,family=family,scene_count=len(by_scene),model_scene_count=len(cases),target=c['target'])
                        # Each persisted stage is immutable; interrupted uncommitted work may rerun.
                        if (job/'init.npy').exists(): init = read(job/'init.npy')
                        else:
                            begin=time.perf_counter(); progress_path=job/'init-progress.npy'
                            progress=read(progress_path) if progress_path.exists() else {}
                            ledger=progress.get('ledger',[]);entries=[]
                            prior_wall=sum(row.get('wall_s',0) for row in ledger)
                            for row in ledger:
                                chunk=job/row.get('chunk',f"init-round{row['round']}-deficit{row['deficit']}.npy")
                                if row['status']=='PASS' and chunk.exists():entries.extend(read(chunk)['entries'])
                            seen={x['fingerprint'] for x in entries};counts={key:0 for key in by_scene}
                            for entry in entries:counts[entry['scene_id']]+=1
                            for round_id in range(c.get('max_rounds',3)):
                                if stop_init():break
                                missing=pending_scenes(counts,ledger,round_id,c['target'],c.get('max_rounds',3))
                                if not missing:continue
                                # Bucket equal deficits: sample only unfinished combinations and cap to deficit.
                                for deficit in sorted({c['target']-counts[key] for key in missing}):
                                    if stop_init():break
                                    selected=[key for key in missing if c['target']-counts[key]==deficit]
                                    chunk_name=f'init-round{round_id}-deficit{deficit}-attempt{len(ledger):04d}.npy'
                                    bucket={'chunk':chunk_name,'round':round_id,'scenes':selected,'deficit':deficit,'status':'UNKNOWN','audit_root':os.environ.get('DEXONOMY_AUDIT_ROOT'),'selected_hard':0,'duplicates':0,'accepted':{key:0 for key in selected}}
                                    ledger.append(bucket)
                                    atomic(progress_path,{'ledger':ledger})
                                    bucket_started=time.perf_counter();entry_start=len(entries)
                                    collected=[]
                                    def collect(scene, _, values):
                                        collected.extend(dict(x,scene_id=scene) for x in values)
                                    cfg=_compose_cfg(base_init+[f'tmpl_name={family}',f'op.filter.general.n_final={deficit}', 'op.object.cfg_path='+json.dumps([by_scene[key]['grasp_scene'] for key in selected])])
                                    event('vnext_init_bucket',tag=tag,round=round_id,scenes=selected,deficit=deficit)
                                    operate_init(cfg,scene_epoch_data_callback=collect,persist_results=False,shared_template_holder=holder)
                                    accepted={key:0 for key in selected}; duplicate=0
                                    for entry in collected:
                                        data=entry['data']; key=entry['scene_id']
                                        assert data['audit_init_hard_all'] and data['tmpl_name']==family
                                        q=np.asarray(data['grasp_qpos'],dtype=np.float64).copy().reshape(-1)
                                        q[3:7]/=np.linalg.norm(q[3:7]);nonzero=np.flatnonzero(np.abs(q[3:7])>1e-12)
                                        if len(nonzero) and q[3+nonzero[0]]<0:q[3:7]*=-1
                                        q=np.round(q,7);q[q==0]=0
                                        fingerprint=digest([key,family,q.tolist()])
                                        if fingerprint in seen:duplicate+=1;continue
                                        seen.add(fingerprint)
                                        assert counts[key]<c['target']
                                        entry.update(id=dataset_id+'/'+tag+'/'+fingerprint,candidate_id=dataset_id+'/'+tag+'/'+fingerprint,round=round_id,fingerprint=fingerprint)
                                        entries.append(entry);counts[key]+=1;accepted[key]+=1
                                    bucket.update(status='PASS',selected_hard=len(collected),duplicates=duplicate,accepted=accepted,wall_s=time.perf_counter()-bucket_started)
                                    if len(entries)>entry_start:atomic(job/chunk_name,{'entries':entries[entry_start:]})
                                    atomic(progress_path,{'ledger':ledger})
                            init={'entries':entries,'counts':counts,'ledger':ledger,'wall_s':prior_wall+time.perf_counter()-begin,'interrupted_wall_unknown':any(row['status']=='UNKNOWN' for row in ledger),'max_rounds':c.get('max_rounds',3),'target':c['target'],'zero':sum(n==0 for n in counts.values()),'short':sum(n<c['target'] for n in counts.values())}
                            atomic(job/'init.npy',init)
                        checkpoint(tag+'/init')
                        release_stage_transient('init_to_refine')
                        entries=init['entries']; prep=time.perf_counter()
                        policy=c.get('simulation_error_policy',{})
                        if c.get('isolate_eval_nonfinite', False) and nonfinite_stop_reason(state['rows'], policy):
                            raise RuntimeError(nonfinite_stop_reason(state['rows'], policy))
                        prior_errors=sum(r.get('simulation_error',0) for r in state['rows'])
                        refine_error_ids=set()
                        if c.get('isolate_refine_nonfinite',False):
                            consecutive=0
                            for row in reversed(state['rows']):
                                if not row.get('refine_simulation_error',0):break
                                consecutive+=1
                            if consecutive>=policy.get('max_consecutive_nonfinite_jobs',2):raise RuntimeError('Consecutive nonfinite jobs limit reached')
                            if prior_errors>=policy.get('max_per_rank',100):raise RuntimeError('Cumulative simulation error limit reached')
                        with open_dict(refine_cfg):
                            refine_cfg.isolate_refine_nonfinite=c.get('isolate_refine_nonfinite',False)
                            refine_cfg.refine_simulation_error_root=str(job/'refine-simulation-errors')
                            refine_cfg.simulation_error_limit=min(policy.get('max_per_rank',100)-prior_errors,policy.get('max_per_job',4),max(1,int(len(entries)*policy.get('max_fraction_per_job',.01))))
                        if group in caches:
                            caches[group][1].release_execution()
                        from dexonomy.sim.data_workspace import clear
                        clear()
                        if entries and group not in caches and not ((job/'refine.npy').exists() and (job/'eval.npy').exists()):
                            caches[group]=(MixedRefinementCache(refine_cfg,[x['grasp_scene'] for x in cases]),batcher_type(eval_cfg,world_batch_size=eval_batch,scene_paths=[x['stability_scene'] for x in cases]))
                            event('vnext_group_prepare',tag=tag,group=group,wall_s=time.perf_counter()-prep,refine=caches[group][0].cache_stats,eval=caches[group][1].cache_stats)
                        preparation=time.perf_counter()-prep
                        if (job/'refine.npy').exists(): refined=read(job/'refine.npy')
                        else:
                            begin=time.perf_counter(); kept=[]; labels=[]
                            for offset in range(0,len(entries),refine_batch):
                                batch=entries[offset:offset+refine_batch]
                                results=refine_grasp_batch_mixed([x['data'] for x in batch],refine_cfg,source_names=[x['id'] for x in batch],world_batch_size=refine_batch,model_cache=caches[group][0],simulation_error_ids=refine_error_ids)
                                assert len(results)==len(batch)
                                for entry,result in zip(batch,results):
                                    labels.append({'id':entry['id'],'success':result['success'],'metrics':result['metrics']})
                                    if result['success']:kept.append(dict(entry,data=result['output_data']))
                                release_warp_scene_memory('cuda:0')
                            refined={'audit_root':os.environ.get('DEXONOMY_AUDIT_ROOT'),'entries':kept,'labels':labels,'wall_s':time.perf_counter()-begin};atomic(job/'refine.npy',refined)
                        successful(c,state,group,family,'refine',len(entries))
                        checkpoint(tag+'/refine')
                        release_stage_transient('refine_to_eval')
                        if (job/'eval.npy').exists(): evaluated=read(job/'eval.npy')
                        else:
                            begin=time.perf_counter(); results=[]
                            event('night_stage_start',stage='eval',tag=tag)
                            if refined['entries']:
                                e=caches[group][1]; e.closed=False
                                e.simulation_error_ids=set()
                                policy=c.get('simulation_error_policy',{})
                                prior_errors=sum(r.get('simulation_error',0) for r in state['rows'])+sum(x['metrics'].get('type')=='simulation_error' for x in refined['labels'])
                                if c.get('isolate_eval_epa',False) or c.get('isolate_eval_nonfinite',False):assert prior_errors<policy.get('max_per_rank',100), 'Cumulative simulation error limit reached'
                                with open_dict(eval_cfg):
                                    eval_cfg.isolate_eval_epa=c.get('isolate_eval_epa',False)
                                    eval_cfg.simulation_error_root=str(job/'simulation-errors')
                                    eval_cfg.simulation_error_limit=min(policy.get('max_per_rank',100)-prior_errors,policy.get('max_per_job',4)-sum(x['metrics'].get('type')=='simulation_error' for x in refined['labels']),max(1,int(len(refined['entries'])*policy.get('max_fraction_per_job',.01))))
                                prepared=[dict(x['data'],scene_path=by_scene[x['scene_id']]['stability_scene'],pipeline_stage='stability_candidate') for x in refined['entries']]
                                report=e.push(prepared,source_names=[x['id'] for x in refined['entries']]);tail=e.flush();results=report['results']+tail['results']
                                assert not e.pending
                            assert len(results)==len(refined['entries']) and len({x['source_name'] for x in results})==len(results)
                            evaluated={'audit_root':os.environ.get('DEXONOMY_AUDIT_ROOT'),'results':results,'wall_s':time.perf_counter()-begin};atomic(job/'eval.npy',evaluated)
                        successful(c,state,group,family,'eval',len(refined['entries']))
                        checkpoint(tag+'/eval')
                        lookup={x['id']:x for x in refined['entries']}
                        feedback_path=job/'feedback.npy'
                        if feedback_path.exists():
                            bundle=read(feedback_path);feedback=bundle['lineage']
                        else:
                            feedback=[];prepared_feedback=[]
                            vis=MuJoCo_VisEnv(hand_cfg=HandCfg(xml_path=loader.xml_path)) if evaluated['results'] else None
                            for result in evaluated['results']:
                                if not result['success']:continue
                                entry=lookup[result['source_name']];data=entry['data'];target=out/'success'/(entry['id']+'.npy')
                                if not target.exists():atomic(target,dict(result['output_data'],provenance={'candidate_id':entry['id'],'scene_id':entry['scene_id'],'template_source':data.get('template_source'),'noslip_iterations':0}))
                                if float(np.asarray(data['n_evo']).reshape(-1)[0])<=2:
                                    prepared_feedback.append(loader._prepare_data(vis,data,entry['id']))
                                    feedback.append({'id':entry['id'],'parent':data.get('template_source'),'family':family,'accepted':None})
                            bundle={'lineage':feedback,'prepared':prepared_feedback}
                            atomic(feedback_path,bundle)
                        checkpoint(tag+'/feedback_prepared')
                        outcomes = apply_feedback(state, loader, family, tag, feedback,
                            bundle['prepared'], feedback_path, out, FamilyPoolLoader)
                        row = completion_row(tag, entries, refined, evaluated, init, preparation, outcomes)
                        record_completion(state, tag, row, loader)
                        capacity_completed(c, state, tag, len(entries))
                        checkpoint(tag+'/complete');event('vnext_job_complete',**row);print(row,flush=True)
                        if c.get('isolate_eval_nonfinite',False) and nonfinite_stop_reason(state['rows'],policy):
                            raise RuntimeError(nonfinite_stop_reason(state['rows'],policy))
                        if c.get('stop_after_jobs') and len(state['completed'])>=c['stop_after_jobs']:raise SystemExit(75)
                        gc.collect();release_warp_scene_memory('cuda:0');torch.cuda.empty_cache()
                if c.get('resident_groups',1)==1:
                    for refine_cache, _ in caches.values():refine_cache.close_traditional_qp_pool()
                    refine_cache=None;_=None;e=None;caches.clear();release_transient()
            state['next_tour']=tour+1
            atomic(statefile,state)
    except DrainComplete:pass
    finally:
        for refine_cache, _ in caches.values():refine_cache.close_traditional_qp_pool()
        loader.stop()
    (out/'summary.json').write_text(json.dumps({'status':'PASS','completion':'DRAINED' if draining else 'COMPLETE','schema':c['checkpoint_schema'],'invocation_wall_s':time.perf_counter()-started,'rows':state['rows'],'resume':'stage transactions; interrupted uncommitted stages may resample; not bitwise RNG continuation'},default=serial,indent=2))

if __name__=='__main__':main()
