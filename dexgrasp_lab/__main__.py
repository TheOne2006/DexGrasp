"""JSON command interface for people and future experiment agents."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

from .config import load_config
from .data import SPLITS, digest, load_manifest, normalize_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    sub.add_parser("list-models")
    p = sub.add_parser("reference-plan")
    for name in ("profile", "python", "data-root", "output-root"):
        p.add_argument(f"--{name}", required=True)
    p = sub.add_parser("prepare-kpgrasp")
    p.add_argument("--root", required=True); p.add_argument("--output", required=True)
    p.add_argument("--val-fraction", type=float, default=0.1); p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("prepare-catalog")
    p.add_argument("--catalog", required=True); p.add_argument("--output", required=True)
    p.add_argument("--val-fraction", type=float, default=0.1); p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("convert-qpos")
    for name in ("source", "cloud", "output", "kpgrasp-root", "joint-order"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--scale", type=float, default=1.0)
    p = sub.add_parser("audit-data")
    p.add_argument("--manifest", required=True); p.add_argument("--deep", action="store_true")
    for command in ("plan", "train"):
        p = sub.add_parser(command); p.add_argument("--config", required=True)
        if command == "train":
            p.add_argument("--execute", action="store_true"); p.add_argument("--resume")
    p = sub.add_parser("sample")
    for name in ("checkpoint", "manifest", "output"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--steps", type=int); p.add_argument("--candidates", type=int)
    p.add_argument("--device", default="cpu"); p.add_argument("--split", type=normalize_split, choices=["val", "test"], default="test")
    p.add_argument("--execute", action="store_true")
    p = sub.add_parser("test", aliases=["metrics"]); p.add_argument("--samples", required=True)
    p = sub.add_parser("ik")
    for name in ("samples", "output", "kpgrasp-root"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--execute", action="store_true")
    p = sub.add_parser("benchmark-plan")
    for name in ("root", "python", "samples", "output", "name"):
        p.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            result = {"python": sys.executable, "version": sys.version,
                      "dependencies": {x: importlib.util.find_spec(x) is not None for x in
                                       ["torch", "numpy", "h5py", "mujoco", "mink", "hydra", "MinkowskiEngine", "pytorch3d"]},
                      "training_requires_isaac_lab": False}
        elif args.command == "reference-plan":
            from .references import plan
            result = plan(args.profile, args.python, args.data_root, args.output_root)
        elif args.command == "list-models":
            result = {"native_architectures": ["pointnet_mlp", "pointnet_transformer"],
                      "native_objectives": ["flow_matching", "diffusion"],
                      "representation": "shadow28_xyz_m_3phase",
                      "official_references": json.loads((Path(__file__).parents[1] / "configs/references.json").read_text())}
        elif args.command == "prepare-kpgrasp":
            from .data import build_kp_manifest
            data = build_kp_manifest(args.root, args.output, args.val_fraction, args.seed)
            result = {"status": "prepared", "records": len(data["records"]), "manifest": args.output}
        elif args.command == "prepare-catalog":
            from .prepare import catalog_manifest
            data = catalog_manifest(args.catalog, args.output, args.val_fraction, args.seed)
            result = {"status": "prepared", "records": len(data["records"]), "manifest": args.output}
        elif args.command == "convert-qpos":
            from .prepare import convert_qpos
            result = convert_qpos(args.source, args.cloud, args.output, args.kpgrasp_root, args.joint_order, args.scale)
        elif args.command == "audit-data":
            data = load_manifest(args.manifest)
            if args.deep:
                from .data import validate_records
                validate_records(data["records"])
            result = {"status": "passed", "sha256": digest(args.manifest),
                      "split_objects": {s: len({r["object_id"] for r in data["records"] if r["split"] == s})
                                        for s in SPLITS},
                      "checks": "shapes/bounds and one finite sample per record" if args.deep else "manifest, paths, leakage"}
        elif args.command in ("plan", "train"):
            cfg = load_config(args.config)
            blockers = []
            try:
                load_manifest(cfg["manifest"])
            except (OSError, ValueError, KeyError) as exc:
                blockers.append(str(exc))
            result = {"status": "blocked" if blockers else "planned", "config": cfg, "blockers": blockers,
                      "effective_batch_size": cfg["batch_size"] * cfg["accumulation"],
                      "optimizer_updates": cfg["max_steps"], "training_started": False}
            if args.command == "train" and args.execute:
                if blockers:
                    raise ValueError("; ".join(blockers))
                from .train import run
                result = run(cfg, args.resume)
        elif args.command == "sample":
            if not args.execute:
                result = {"status": "planned", "training_started": False, "arguments": vars(args)}
            else:
                from .test import generate
                result = generate(args.checkpoint, args.manifest, args.output, args.steps,
                                  args.candidates, args.device, args.split)
        elif args.command in ("test", "metrics"):
            from .test import metrics
            result = metrics(args.samples)
        elif args.command == "ik":
            if not args.execute:
                result = {"status": "planned", "arguments": vars(args)}
            else:
                from .test import reconstruct
                result = reconstruct(args.samples, args.output, args.kpgrasp_root)
        elif args.command == "benchmark-plan":
            from .test import benchmark_plan
            result = benchmark_plan(args.root, args.python, args.samples, args.output, args.name)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 2 if result.get("status") == "blocked" else 0
    except (ValueError, KeyError, OSError, ImportError, RuntimeError, FloatingPointError) as exc:
        print(json.dumps({"status": "error", "type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
