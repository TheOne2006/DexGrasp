"""Pure bookkeeping extracted from the worker, with no physics or disk writes.

The worker owns atomic writes and checkpoint order. These helpers mutate only
in-memory state/loader bookkeeping; committing it remains the caller's job.
"""
import copy
import hashlib


def checkpoint_artifacts(state, out, stage):
    """Collect the original stage/error hashes before the worker commits state."""
    tag=stage.rsplit('/',1)[0]
    for name in ['init.npy','refine.npy','eval.npy','feedback.npy']:
        path=out/'jobs'/tag/name;rel=str(path.relative_to(out))
        if path.exists() and rel not in state.setdefault('artifacts',{}):
            state['artifacts'][rel]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path in list((out/'jobs'/tag/'simulation-errors').glob('*.npy'))+list((out/'jobs'/tag/'refine-simulation-errors').glob('*.npy')):
        state.setdefault('artifacts',{})[str(path.relative_to(out))]=hashlib.sha256(path.read_bytes()).hexdigest()


def apply_feedback(state, loader, family, tag, feedback, prepared_items, feedback_path, out, family_loader_type):
    """Apply prepared feedback in original order, including rejected statistics."""
    outcomes=[]
    for feedback_index,(item,prepared) in enumerate(zip(feedback,prepared_items)):
        assert item['id'] not in loader.feedback_seen_ids, 'Feedback must commit once'
        accepted=loader._append_hand_data(prepared) is not False
        outcome='accepted' if accepted else 'rejected'
        outcomes.append(dict(item,accepted=accepted))
        if accepted:
            loader.feedback_seen_ids.add(item['id'])
            state['family_archive'].setdefault(family,[]).append({'id':item['id'],'path':str(feedback_path.relative_to(out)),'index':feedback_index})
        loader._record_feedback_stat('received')
        loader._record_feedback_stat(outcome)
        if isinstance(loader,family_loader_type):
            loader._record_family_feedback(family, 'received')
            loader._record_family_feedback(family, outcome)
    state.setdefault('feedback_outcomes',{})[tag]=outcomes
    state['feedback_seen_ids']=sorted(loader.feedback_seen_ids)
    return outcomes


def completion_row(tag, entries, refined, evaluated, init, preparation, outcomes):
    """Keep success/error/feedback denominators and timing fields unchanged."""
    return {
        'tag': tag,
        'init': len(entries),
        'refined': len(refined['entries']),
        'stable': sum(bool(x['success']) for x in evaluated['results']),
        'simulation_error': (
            sum(x['metrics'].get('type') == 'simulation_error' for x in evaluated['results'])
            + sum(x['metrics'].get('type') == 'simulation_error' for x in refined['labels'])
        ),
        'nonfinite': sum(x['metrics'].get('type') == 'simulation_error' and x['metrics'].get('error') == 'nonfinite'
                         for x in evaluated['results'] + refined['labels']),
        'refine_simulation_error': sum(
            x['metrics'].get('type') == 'simulation_error' for x in refined['labels']
        ),
        'init_wall_s': init['wall_s'],
        'refine_wall_s': refined['wall_s'],
        'eval_wall_s': evaluated['wall_s'],
        'prepare_s': preparation,
        'zero': init['zero'],
        'short': init['short'],
        'feedback': sum(x['accepted'] for x in outcomes),
        'feedback_rejected': sum(not x['accepted'] for x in outcomes),
    }



def record_completion(state, tag, row, loader):
    """Snapshot memory immediately before the worker's atomic complete write."""
    state['completed'].append(tag)
    state['rows'].append(row)
    state['template_buffer'] = copy.deepcopy(loader.data_buffer)
    state['template_stats'] = copy.deepcopy(loader.feedback_stats)
