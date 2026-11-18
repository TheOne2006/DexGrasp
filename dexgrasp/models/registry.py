"""Model families pair an architecture with its sampler on normalized conditions."""
from dataclasses import dataclass
from typing import Callable

from .config import ModelConfig
from .execution import configure_keypoint_execution
from .keypoint import KeypointModel
from .sampling import sample_keypoints


@dataclass(frozen=True)
class ModelBranch:
    config_type: type
    build: Callable
    sample: Callable
    default_steps: dict[str, int]
    requires_minkowski: bool = False
    configure: Callable | None = None


MODEL_BRANCHES = {
    'keypoint_flow': ModelBranch(ModelConfig, KeypointModel, sample_keypoints,
                                 {'fm': 10, 'flowmap': 1}, requires_minkowski=True,
                                 configure=configure_keypoint_execution),
}


def model_branch(family):
    try:
        return MODEL_BRANCHES[family]
    except KeyError:
        raise ValueError(f'unknown model family: {family!r}') from None


def config_from_settings(settings, hand_spec=None):
    config_type = model_branch(settings.get('family', 'keypoint_flow')).config_type
    if hand_spec is not None and 'hand_reference_length_m' in config_type.__dataclass_fields__:
        settings = dict(settings, hand_reference_length_m=hand_spec['reference_length_m'])
    return config_type(**settings)


def build_model(cfg, method, *, encoder_factory=None):
    branch = model_branch(cfg.family)
    if method not in branch.default_steps:
        raise ValueError(f'{cfg.family} does not support {method}')
    options = {} if encoder_factory is None else {'encoder_factory': encoder_factory}
    return branch.build(cfg, method, **options)


def sample_model(model, condition, *, steps=None, candidates=1, **options):
    branch = model_branch(model.cfg.family)
    if steps is None:
        steps = branch.default_steps[model.method]
    return branch.sample(model, condition, steps=steps, candidates=candidates, **options)
