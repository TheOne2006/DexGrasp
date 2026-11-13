"""Replay frozen strict Init candidates without re-sampling or modifying old outputs."""
import argparse,hashlib,json,time,os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from omegaconf import open_dict
from dexonomy.upstream_stream import _compose_cfg,_common_overrides,_grasp_overrides
from dexonomy.op.gen_grasp_warp_resident import MixedRefinementCache,refine_grasp_batch_mixed
from dexonomy.op.eval_force_closure_warp_mixed import MixedForceClosureBatcher
from dexonomy.sim.data_workspace import clear
from night_worker import atomic


def main():
    p=argparse.ArgumentParser();p.add_argument('config',type=Path);p.add_argument('output',type=Path);a=p.parse_args();c=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=True)
    frozen=Path(c['replay_init']);assert hashlib.sha256(frozen.read_bytes()).hexdigest()==c['replay_init_sha256']
    init=np.load(frozen,allow_pickle=True).item();entries=init['entries'];assert all(e['data']['audit_init_hard_all'] for e in entries)
    cases=json.loads(Path(c['groups'][0]['index']).read_text())['cases'];by={x['scene_id']:x for x in cases}
    args=SimpleNamespace(hand='sharpa_v2',asset_root=c['asset_root'],save_root=str(a.output),workers=1,pregrasp_cdist=.045,pregrasp_ho_thre=.04,warp_device='cuda:0',warp_world_batch_size=c['refine_batch'])
    common=lambda op:_common_overrides(args,op,'capacity-replay')+['obj_mass=0.1']
    rc=_compose_cfg(common('grasp')+_grasp_overrides(args)+['+refine_qp_backend=cpu',f'+refine_ccd_multiplier={c["refine_ccd_multiplier"]}',f'+refine_min_execution_slots={c["refine_min_execution_slots"]}',f'+refine_small_batch_limit={c["refine_small_batch_limit"]}',f'+refine_small_batch_capacity={c["refine_small_batch_capacity"]}'])
    ec=_compose_cfg(common('eval')+['physics_profile=native','op.reuse_fc_preamble=false','+warp_device=cuda:0','+terminal_stability=True','op.noslip_iterations=0'])
    with open_dict(rc):rc.refine_njmax=c['refine_njmax'];rc.refine_cpu_qp_workers=int(c.get('refine_cpu_qp_workers',len(os.sched_getaffinity(0))));rc.refine_cpu_qp_executor='thread'
    with open_dict(ec):ec.eval_njmax=c['eval_njmax']
    with open_dict(rc):
        rc.refine_nconmax=c.get('refine_nconmax',0);rc.refine_nccdmax=c.get('refine_nccdmax',0)
    with open_dict(ec):
        ec.eval_nconmax=c.get('eval_nconmax',0);ec.eval_nccdmax=c.get('eval_nccdmax',0)
    tabletop=c.get('checkpoint_schema')=='tabletop-single-gpu/v1'
    batcher=MixedForceClosureBatcher
    if tabletop:
        from dexonomy.tabletop_batcher import TabletopSlideFCBatcher
        batcher=TabletopSlideFCBatcher
        with open_dict(ec):ec.tabletop_output=str(a.output/'tabletop')
    cache=MixedRefinementCache(rc,[x['grasp_scene'] for x in cases]);ev=batcher(ec,world_batch_size=c['eval_batch'],scene_paths=[x['stability_scene'] for x in cases]);start=time.monotonic()
    atomic(a.output/'input.npy',init)
    try:
        results=[]
        for offset in range(0,len(entries),c['refine_batch']):
            part=entries[offset:offset+c['refine_batch']]
            results.extend(refine_grasp_batch_mixed([x['data'] for x in part],rc,source_names=[x['id'] for x in part],world_batch_size=c['refine_batch'],model_cache=cache))
        assert len(results)==len(entries);atomic(a.output/'refine.npy',{'results':results})
        kept=[dict(e,data=r['output_data']) for e,r in zip(entries,results) if r['success']];clear()
        output=[]
        if kept:
            data=[dict(x['data'],scene_path=by[x['scene_id']]['stability_scene'],pipeline_stage='stability_candidate') for x in kept]
            output=ev.push(data,source_names=[x['id'] for x in kept])['results']+ev.flush()['results']
        assert len(output)==len(kept);atomic(a.output/'eval.npy',{'results':output})
        summary=dict(status='PASS',strict_init=len(entries),refined=len(kept),stable=sum(bool(x['success']) for x in output),six_direction_worlds=6*len(output),refine_labels=[dict(id=e['id'],scene=e['scene_id'],success=r['success']) for e,r in zip(entries,results)],wall_s=time.monotonic()-start,refine_njmax=c['refine_njmax'],eval_njmax=c['eval_njmax'],input_sha256=c['replay_init_sha256'])
        if tabletop:
            summary['slide_pass']=ev.last_report['counts']['slide_pass'] if output else 0
            summary['six_direction_worlds']=ev.last_report['counts']['direction_tasks'] if output else 0
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    finally:cache.close_traditional_qp_pool()

if __name__=='__main__':main()
