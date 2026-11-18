"""Apply the execution settings supported by the selected model family."""
from ..models.registry import model_branch


def configure_execution(model, cfg):
    configure = model_branch(model.cfg.family).configure
    if configure is not None:
        configure(model, cfg)


def configure_attention_precision(model):
    configure_execution(model, None)
