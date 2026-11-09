"""Inspect official execution plans without importing their native dependencies."""
import json
from pathlib import Path

PROFILES = {
    "kpgrasp_fm": ("KPGrasp", "Final_base"),
    "kpgrasp_flowmap": ("KPGrasp", "Final_lsd3"),
    "dexlearn_diffusion": ("DexLearn", "diffusion"),
    "dexlearn_nflow": ("DexLearn", "nflow"),
    "dexgraspnet2_diffusion": ("DexGraspNet2", "configs/network/train_dex_ours.yaml"),
}


def plan(profile, python, data_root, output_root):
    if profile not in PROFILES:
        raise ValueError(f"unknown reference profile: {profile}")
    name, experiment = PROFILES[profile]
    code_root = Path(__file__).resolve().parents[1]
    metadata = next(r for r in json.loads((code_root / "configs/references.json").read_text()) if r["name"] == name)
    root = (code_root / "configs" / metadata["path"]).resolve()
    data_root, output_root = str(Path(data_root).resolve()), str(Path(output_root).resolve())
    limitations = ["Runtime and native extensions not validated; no command executed."]
    env = {"CUDA_VISIBLE_DEVICES": "0", "WANDB_MODE": "disabled"}
    if name == "KPGrasp":
        argv = [python, "src/dexgrasp/train.py", f"+experiment={experiment}",
                "algo.batch_size=8", "data.num_workers=0", "data.pre_load=false",
                f"data_folder={data_root}", f"output_folder={output_root}", "wandb.mode=disabled"]
        limitations += ["Upstream eval uses test.json: isolate validation before selecting checkpoints.",
                        "pre_load=false still loads entire grasp shards on first access.",
                        "Final_base depth=8; Final_lsd3 depth=16. Not an architecture-controlled comparison."]
    elif name == "DexLearn":
        argv = [python, "-m", "dexlearn.train", f"algo={experiment}", "data=dexonomy_shadow",
                f"exp_name={profile}", "algo.batch_size=8", "data.num_workers=0",
                f"data_folder={data_root}", f"output_folder={output_root}", "resume=false", "wandb.mode=disabled"]
        limitations += ["Upstream eval uses test.json: isolate validation before selecting checkpoints.",
                        "Default grasp_type_lst selects only 1_Large_Diameter; taxonomy scope must be explicit."]
    else:
        argv = [python, "src/train.py", "--exp_name", profile, "--yaml", experiment]
        limitations += ["Upstream uses checkout-relative data/ and experiments/: data/output arguments here are NOT applied.",
                        "Create a separate runnable checkout and data link/config before execution.",
                        "Leap Hand and cluttered scenes are not comparable to Shadow Hand single-object metrics."]
    return {"status": "reference_plan_only", "profile": profile, "cwd": str(root),
            "argv": argv, "env": env, "source": metadata, "limitations": limitations}
