"""Read back stage transactions, strict candidates, saved successes and provenance."""
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-interrupted-init',action='store_true');a=p.parse_args();state=np.load(a.checkpoint/'state.npy',allow_pickle=True).item();bad=[];totals=collections.Counter();ids=set();families=set();scenes=set();sources=collections.Counter();historical=set(state.get('historical_template_ids',[]));prior_feedback=set(historical);evolved_sources=set();attempted_scenes=set();attempted_families=set()
for tag in state['completed']:
    job=a.checkpoint/'jobs'/tag
    init=np.load(job/'init.npy',allow_pickle=True).item();ref=np.load(job/'refine.npy',allow_pickle=True).item();ev=np.load(job/'eval.npy',allow_pickle=True).item();feedback=np.load(job/'feedback.npy',allow_pickle=True).item()
    attempted_scenes.update(init['counts']);attempted_families.add(tag.split('-e',1)[1].split('-',1)[1])
    entries=init['entries'];current={x['id']:x for x in entries};totals['init']+=len(entries);totals['raw_round_scene_visits']+=sum(len(x['scenes']) for x in init['ledger'])
    if len(current)!=len(entries) or ids.intersection(current):bad.append([tag,'duplicate_candidate_id'])
    ids.update(current)
    counts=collections.Counter(x['scene_id'] for x in entries)
    if any(counts[key]!=n or n>init['target'] for key,n in init['counts'].items()):bad.append([tag,'quota_counts'])
    roundsets=collections.defaultdict(list);attempts=collections.Counter()
    for row in init['ledger']:
        attempts.update(row['scenes'])
        if a.allow_interrupted_init and row['status']=='UNKNOWN':
            totals['unknown_round_scene_visits']+=len(row['scenes']);continue
        for scene in row['scenes']:roundsets[scene].append(row['round'])
    if any(len(v)!=len(set(v)) or max(v)>=init.get('max_rounds',3) for v in roundsets.values()):bad.append([tag,'sampling_round_budget'])
    if a.allow_interrupted_init and any(n>init.get('max_rounds',3) for n in attempts.values()):bad.append([tag,'interrupted_attempt_budget'])
    for entry in entries:
        data=entry['data'];families.add(data['tmpl_name']);scenes.add(entry['scene_id']);sources[str(data.get('template_source'))]+=1
        source=data.get('template_source')
        if source in prior_feedback:evolved_sources.add(source)
        elif source and not Path(source).is_file():bad.append([tag,'unknown_or_future_template_parent',str(source)])
        if not data['audit_init_hard_all']:bad.append([tag,'failed_init_selected'])
        if not np.isfinite(data['grasp_qpos']).all():bad.append([tag,'nonfinite_init'])
    refined={x['id']:x for x in ref['entries']};labels={x['id']:x['success'] for x in ref['labels']}
    if set(labels)!=set(current) or set(refined)!={key for key,value in labels.items() if value}:bad.append([tag,'refine_coverage'])
    results={x['source_name']:x for x in ev['results']}
    if len(results)!=len(ev['results']) or set(results)!=set(refined):bad.append([tag,'eval_coverage'])
    refine_errors={x['id'] for x in ref['labels'] if x['metrics'].get('type')=='simulation_error'}
    if refine_errors & set(refined):bad.append([tag,'refine_error_entered_eval'])
    if any(x['success'] for x in ref['labels'] if x['id'] in refine_errors):bad.append([tag,'refine_error_success'])
    totals['refine_simulation_error']+=len(refine_errors);totals['simulation_error']+=len(refine_errors)
    totals['refined']+=len(refined);totals['eval_worlds']+=sum(len(x['metrics']['directions']) if 'directions' in x['metrics'] and 'slide' in x['metrics'] else 6 for x in results.values() if x.get('metrics',{}).get('type')!='simulation_error');totals['simulation_error']+=sum(x.get('metrics',{}).get('type')=='simulation_error' for x in results.values());totals['feedback']+=len(feedback['lineage'])
    for key,result in results.items():
        if result['success']:
            totals['stable']+=1;path=a.checkpoint/'success'/(key+'.npy')
            if not path.exists():bad.append([key,'missing_saved_success']);continue
            saved=np.load(path,allow_pickle=True).item()
            if saved['provenance']['candidate_id']!=key:bad.append([key,'provenance_mismatch'])
            for name,value in result['output_data'].items():
                if isinstance(value,np.ndarray) and (name not in saved or not np.array_equal(saved[name],value)):bad.append([key,'saved_output_mismatch',name])
            for name,v in saved.items():
                if isinstance(v,np.ndarray) and v.dtype.kind in 'fci' and not np.isfinite(v).all():bad.append([key,'nonfinite_success',name])
    outcomes=state.get('feedback_outcomes',{}).get(tag,feedback['lineage'])
    totals['feedback_accepted']+=sum(x.get('accepted',False) is True for x in outcomes)
    for item in feedback['lineage']:
        if item['id'] not in results or not results[item['id']]['success']:bad.append([tag,'invalid_feedback'])
    prior_feedback.update(x['id'] for x in outcomes if x.get('accepted') is True)
if 'feedback_seen_ids' in state and (set(state['feedback_seen_ids'])!=prior_feedback or len(state['feedback_seen_ids'])!=len(prior_feedback)):bad.append(['feedback_committed_ids'])
if 'feedback_outcomes' in state and state['template_stats']['accepted']!=len(prior_feedback-historical):bad.append(['feedback_accepted_count'])
paths=state['template_buffer']['hand_path']
if len(paths)!=len(set(paths)) or any(len(v)!=len(paths) for v in state['template_buffer'].values()):bad.append(['template_buffer_duplicates_or_shape'])
report={'status':'FAIL' if bad else 'PASS','bad':bad,'totals':dict(totals),'jobs':len(state['completed']),'planned_jobs':state.get('planned_jobs'),'completion':'COMPLETE' if state.get('planned_jobs')==len(state['completed']) else 'UNKNOWN_OR_PARTIAL','attempted_scenes':len(attempted_scenes),'attempted_objects':len({x.rsplit('_scale',1)[0] for x in attempted_scenes}) if state.get('schema') not in ('tabletop-single-gpu/v1','tabletop-night/v1') else None,'attempted_families':sorted(attempted_families),'evolved_templates_actually_sampled':len(evolved_sources),'historical_templates_actually_sampled':len(evolved_sources & historical),'new_feedback_actually_sampled':len(evolved_sources-historical),'imported_historical_templates':len(historical),'scenes':len(scenes),'objects':len({x.rsplit('_scale',1)[0] for x in scenes}) if state.get('schema') not in ('tabletop-single-gpu/v1','tabletop-night/v1') else None,'families':sorted(families),'sampled_template_sources':dict(sources),'boundary':'Stage artifact accounting. Six-direction physical checkpoints require audit verification separately.'}
a.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='sampled_template_sources'}));raise SystemExit(bool(bad))
