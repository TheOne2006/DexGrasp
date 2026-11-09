"""Single-device training. Importing this module never starts a run."""
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .data import batch, digest, load_manifest, validate_records, write_json
from .methods import loss
from .models import GraspModel


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_checkpoint(path, state):
    temporary = Path(str(path) + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def source_state():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
            "source_sha256": {str(p.relative_to(root)): digest(p)
                              for p in sorted((root / "dexgrasp_lab").glob("*.py"))}}


def validate(model, records, cfg):
    # Validation repeats identical object/grasp/noise choices and restores training RNGs.
    device = torch.device(cfg["device"])
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    previous_mode = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(cfg["seed"] + 1000)
            rng = np.random.default_rng(cfg["seed"] + 1000)
            selected = list(range(min(len(records), cfg["val_objects"])))
            for start in range(0, len(selected), cfg["batch_size"]):
                ids = selected[start:start + cfg["batch_size"]]
                pc, kp, _ = batch(records, ids, rng, cfg["num_points"], cfg["scale_m"], device)
                value = loss(model, pc, kp, cfg["objective"])
                total += value.item() * len(ids)
                count += len(ids)
    finally:
        model.train(previous_mode)
    if count == 0 or not np.isfinite(total):
        raise ValueError("empty/nonfinite validation")
    return total / count


def run(cfg, resume=None):
    manifest = load_manifest(cfg["manifest"])
    validate_records(manifest["records"])
    data_hash = digest(cfg["manifest"])
    run_dir, ckpt_dir = Path(cfg["run_dir"]), Path(cfg["checkpoint_dir"])
    if run_dir == ckpt_dir:
        raise ValueError("run and checkpoint directories must be distinct")
    if not resume and any(p.exists() and any(p.iterdir()) for p in (run_dir, ckpt_dir)):
        raise FileExistsError("run/checkpoint directory is not empty; use a new run ID or --resume")
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in selected interpreter")
    seed_all(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    model = GraspModel(cfg["architecture"], cfg["width"], cfg["depth"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    start, best = 0, float("inf")
    sources = source_state()
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state["config"] != cfg or state["manifest_sha256"] != data_hash:
            raise ValueError("resume requires identical config and manifest")
        if state["source"]["source_sha256"] != sources["source_sha256"]:
            raise ValueError("source changed since checkpoint; use a separate experiment")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, best = state["step"], state["best_val"]
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    if start >= cfg["max_steps"]:
        raise ValueError("checkpoint already reached max_steps")
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        file_facts = {}
        for row in manifest["records"]:
            for key in ("path", "grasp_h5", "point_cloud_h5"):
                if key in row:
                    p = Path(row[key]); stat = p.stat()
                    file_facts[str(p)] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        write_json(run_dir / "provenance.json", {
            "config": cfg, "manifest_sha256": data_hash, "manifest": manifest,
            "source": sources, "torch": torch.__version__, "data_files": file_facts,
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "note": "Data file sizes/mtimes are recorded, not content integrity proofs."})
    train = [r for r in manifest["records"] if r["split"] == "train"]
    val = [r for r in manifest["records"] if r["split"] == "val"]
    # Uniform objects, then uniform records, then uniform grasps within that record.
    groups = {}
    for i, row in enumerate(train):
        groups.setdefault(row["object_id"], []).append(i)
    groups = list(groups.values())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    started = time.monotonic()
    with (run_dir / "metrics.jsonl").open("a") as log:
        for step in range(start + 1, cfg["max_steps"] + 1):
            optimizer.zero_grad(set_to_none=True)
            train_loss = 0.0
            for _ in range(cfg["accumulation"]):
                ids = [rng.choice(groups[int(rng.integers(len(groups)))]) for _ in range(cfg["batch_size"])]
                pc, kp, _ = batch(train, ids, rng, cfg["num_points"], cfg["scale_m"], device)
                value = loss(model, pc, kp, cfg["objective"]) / cfg["accumulation"]
                if not torch.isfinite(value):
                    raise FloatingPointError("nonfinite loss")
                value.backward()
                train_loss += value.item()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            record = {"step": step, "loss": train_loss, "grad_norm": norm.item(),
                      "seconds": time.monotonic() - started}
            improved = False
            if step % cfg["val_every"] == 0 or step == cfg["max_steps"]:
                record["val_loss"] = validate(model, val, cfg)
                improved = record["val_loss"] < best
                best = min(best, record["val_loss"])
            log.write(json.dumps(record, allow_nan=False) + "\n"); log.flush()
            if improved or step % cfg["save_every"] == 0 or step == cfg["max_steps"]:
                state = {"schema_version": 1, "step": step, "best_val": best,
                         "config": cfg, "manifest_sha256": data_hash, "source": sources,
                         "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                         "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
                atomic_checkpoint(ckpt_dir / "last.pt", state)
                if improved:
                    atomic_checkpoint(ckpt_dir / "best.pt", state)
    return {"status": "completed", "steps": cfg["max_steps"], "best_val_loss": best,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
