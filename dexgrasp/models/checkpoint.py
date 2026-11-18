"""Rebuild the recorded model family and load its parameters for inference."""
import torch
from .registry import build_model, config_from_settings, model_branch


def load_model(checkpoint, device, *, encoder_factory=None):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(state.get('model_config'), dict) or not state.get('method'):
        raise ValueError('unsupported DexGrasp checkpoint')
    mc = config_from_settings(state['model_config'])
    if mc.checkpoint_schema != state['schema']:
        raise ValueError('checkpoint schema and stage count disagree')
    model = build_model(mc, state['method'], encoder_factory=encoder_factory).to(device)
    model.load_state_dict(state['model'])
    configure = model_branch(mc.family).configure
    if configure is not None:
        configure(model, None)
    model.eval()
    return model, state
