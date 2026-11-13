"""Explicit optimizer/RNG-preserving forks for attention numerical repairs."""
from pathlib import Path
import torch


def validate_recovery(state, cfg, identity, runtime, world, hand_hash, initial_hash):
    """Permit only output paths, the numerical attention backend and a stop guard.

    Source changes are intentional for a recovery fork, recorded in provenance;
    ordinary --resume continues to require byte-identical source and recipe.
    """
    if (state.get('schema') != 'dexgrasp-v1' or state.get('method') != 'fm'
            or state['data_identity'] != identity or state['runtime'] != runtime
            or state['distributed']['world_size'] != world
            or state['hand_spec_sha256'] != hand_hash
            or state['initial_checkpoint_sha256'] != initial_hash):
        raise ValueError('recovery requires identical data/runtime/hand/parent/world size')
    old, new = dict(state['config']), dict(cfg)
    for key in ('run_dir', 'checkpoint_dir'):
        if Path(old[key]).resolve() == Path(new[key]).resolve():
            raise ValueError('recovery requires new output directories')
        old.pop(key); new.pop(key)
    old.pop('abort_grad_norm', None); new.pop('abort_grad_norm', None)
    old_model, new_model = dict(old.pop('model')), dict(new.pop('model'))
    old_model.pop('self_attention_backend', None)
    new_model.pop('self_attention_backend', None)
    if old != new or old_model != new_model:
        raise ValueError('recovery cannot change training recipe or model architecture')
    if state['distributed']['global_batch_size'] != cfg['batch_size']*world*cfg['accumulation']:
        raise ValueError('recovery global batch mismatch')
    if len(state['distributed']['rank_rng']) != world:
        raise ValueError('recovery requires RNG state for every rank')
    for section in ('model', 'optimizer'):
        def check(value):
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise ValueError(f'nonfinite recovery {section} state')
            if isinstance(value, dict):
                for child in value.values(): check(child)
            elif isinstance(value, (list, tuple)):
                for child in value: check(child)
        check(state[section])
