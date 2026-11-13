"""Pure worker routing and configuration assembly; no physics imports."""
from types import SimpleNamespace
from night_inventory import validate_mode
from night_schedule import validate_schedule


def validate_worker_mode(config):
    mode = validate_mode(config)
    assert not config.get('deployment_template_only', False), 'Deployment template cannot launch'
    assert 0 <= config['rank'] < config['world_size']
    if mode == "tabletop" and (config.get("isolate_eval_nonfinite", False) or config.get("eval_failed_world_retirement", False)):
        raise ValueError("Floating Eval recovery/retirement flags are not supported by Tabletop Slide-FC")
    validate_schedule(config)
    return mode


def stage_configs(c, out, tabletop, compose, common_overrides, grasp_overrides, open_dict):
    """Compose the existing stage overrides in their original order.

    Composer/context callbacks come from the caller; this helper does not import
    the runtime, select an interpreter, load assets or initialize a device.
    """
    args = SimpleNamespace(hand='sharpa_v2',asset_root=c['asset_root'],save_root=str(out),workers=c.get('loader_workers',1),
        pregrasp_cdist=.045,pregrasp_ho_thre=.04,warp_device='cuda:0',warp_world_batch_size=c['refine_batch'])
    common = lambda op: common_overrides(args,op,'vnext')+['obj_mass=0.1']
    refine_cfg = compose(common('grasp')+grasp_overrides(args)+['+refine_qp_backend=cpu',f'+refine_ccd_multiplier={c.get("refine_ccd_multiplier",1)}',f'+refine_min_execution_slots={c["refine_min_execution_slots"]}',f'+refine_small_batch_limit={c["refine_small_batch_limit"]}',f'+refine_small_batch_capacity={c["refine_small_batch_capacity"]}'])
    # Explicit opt-in; the production snapshot's default remains serial CPU QP.
    with open_dict(refine_cfg):
        refine_cfg.refine_njmax = c.get('refine_njmax',0)
        refine_cfg.refine_cpu_qp_workers = int(c.get('refine_cpu_qp_workers', 0))
        refine_cfg.refine_cpu_qp_executor = str(c.get('refine_cpu_qp_executor', 'thread'))
    eval_cfg = compose(common('eval')+['physics_profile=native','op.reuse_fc_preamble=false','+warp_device=cuda:0','+terminal_stability=True','op.noslip_iterations=0'])
    with open_dict(eval_cfg):
        eval_cfg.eval_njmax=c.get('eval_njmax',0)
        if not tabletop:
            for option in ('isolate_eval_nonfinite', 'eval_failed_world_retirement'):
                if option in c:
                    eval_cfg[option]=bool(c[option])
        if tabletop:
            eval_cfg.tabletop_output=str(out/'tabletop')
            eval_cfg.tabletop_trace=bool(c.get('tabletop_trace', False))
            eval_cfg.tabletop_trace_candidates=c.get('tabletop_trace_candidates', [])
    base_init = common('init')+['init_gpu=[0]','op.epoch=1',f'op.object.batch_size={c["object_batch"]}', 'op.object.n_cfg=0',f'op.object.n_init_point={c["sample_points"]}',f'op.object.n_init_inplane={c.get("inplane",8)}','op.matcher.opt_step=100','+op.filter.general.strict=true']
    seed_cfg = compose(base_init)
    return refine_cfg, eval_cfg, base_init, seed_cfg
