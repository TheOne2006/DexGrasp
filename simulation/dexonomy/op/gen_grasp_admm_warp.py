"""Explicit experimental entry; production defaults remain traditional."""
import copy
from omegaconf import open_dict
from dexonomy.op.gen_grasp_warp_resident import refine_grasp_batch_mixed


def refine_admm_batch_mixed(candidates,cfg,*,source_names=None,world_batch_size=640,model_cache=None):
    cfg=copy.deepcopy(cfg)
    with open_dict(cfg):cfg.refine_algorithm='admm'
    results=refine_grasp_batch_mixed(candidates,cfg,source_names=source_names,world_batch_size=world_batch_size,model_cache=model_cache)
    failed=[i for i,r in enumerate(results) if r['metrics']['admm']['fallback']]
    if failed:
        fallback=copy.deepcopy(cfg)
        with open_dict(fallback):
            fallback.refine_algorithm='traditional';fallback.op.grasp.step=20;fallback.op.grasp.substep=10
        recovered=refine_grasp_batch_mixed([candidates[i] for i in failed],fallback,
            source_names=None if source_names is None else [source_names[i] for i in failed],world_batch_size=world_batch_size,model_cache=model_cache)
        for i,r in zip(failed,recovered):
            diagnostic=results[i]['metrics']['admm'];prior=results[i]['metrics']
            r['metrics']['admm']=diagnostic;r['metrics']['admm_before_fallback']=prior
            results[i]=r
    return results
