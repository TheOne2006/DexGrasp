"""Global pair diversity of native DexGrasp outputs: Sharpa21 and frozen hand50.

``evaluate_dataset`` reads trusted native output dictionaries and defaults to
exact all-record pairs. ``evaluate_pair_features`` accepts already weighted
descriptors. Neither fits a kernel to the evaluated input or groups its pairs.
"""
from ._version import __version__
from .features import FeatureConfig
from .hand import SharpaHand
from .metrics import evaluate_pair_features
from .evaluator import evaluate_dataset

__all__ = ["evaluate_dataset", "evaluate_pair_features", "FeatureConfig", "SharpaHand", "__version__"]
