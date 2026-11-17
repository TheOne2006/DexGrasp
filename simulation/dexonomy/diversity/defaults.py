"""Self-contained frozen hand50 parameters and their historical provenance."""
from dataclasses import asdict
import hashlib
from importlib.resources import files
import json

from .features import FeatureConfig

PROFILE_RESOURCE = "resources/hand50.json"
PROFILE_RESOURCE_SHA256 = "15cb39493ece33d07dc082ad702f1eba53c92c3beb6bddb0a85e244681ea13bc"


def _config_hash(config):
    text = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def load_profile():
    """Return fixed hand50 config/metadata; never read an experiment directory."""
    raw = files("dexonomy.diversity").joinpath(PROFILE_RESOURCE).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROFILE_RESOURCE_SHA256:
        raise ValueError("Bundled hand50 configuration SHA256 mismatch")
    spec = json.loads(raw)
    if spec["schema"] != "dexonomy-frozen-hand50/v1" or spec["name"] != "hand50":
        raise ValueError("Invalid bundled hand50 definition")
    config = FeatureConfig(**spec["feature_config"])
    if config != FeatureConfig():
        raise ValueError("FeatureConfig defaults disagree with the frozen hand50 definition")
    return config, {
        "name": "hand50", "definition": spec["definition"],
        "feature_config": asdict(config), "feature_config_sha256": _config_hash(config),
        "reference": spec["reference"],
        "configuration_resource": {"resource": PROFILE_RESOURCE, "sha256": PROFILE_RESOURCE_SHA256},
        "fitted_to_evaluation_inputs": False,
        "not_recalibrated_for_current_keypoints": True,
        "resolution_multiplier": 1.0,
    }


def configured_parameters(config=None):
    """Label nondefault explicit parameters as custom, never as hand50."""
    frozen, metadata = load_profile()
    if config is None or config == frozen:
        return frozen, metadata
    if not isinstance(config, FeatureConfig):
        raise ValueError("config must be a FeatureConfig or None")
    return config, {
        "name": "custom", "definition": "Explicit caller parameters; no calibration is inferred.",
        "feature_config": asdict(config), "feature_config_sha256": _config_hash(config),
        "reference": None, "configuration_resource": None,
        "fitted_to_evaluation_inputs": False,
        "not_recalibrated_for_current_keypoints": None,
        "resolution_multiplier": None,
    }
