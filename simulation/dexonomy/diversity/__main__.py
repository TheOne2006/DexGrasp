"""Global Sharpa21 diversity using the bundled frozen hand50 configuration."""
import argparse
import json
from pathlib import Path
import sys

from .evaluator import evaluate_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Trusted native grasp NPY files or generation/success directories")
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--mode", choices=["exact", "sample"], default="exact")
    parser.add_argument("--num-pairs", type=int, help="Total global sample draws B; overrides --pairs-per-grasp")
    parser.add_argument("--pairs-per-grasp", type=int, default=100, help="Fallback sample budget B=c*global N")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--feature-batch-size", type=int, default=1024)
    parser.add_argument("--protocol", choices=["auto", "floating", "tabletop"], default="auto")
    parser.add_argument("--selection", choices=["success", "all"], default="success")
    parser.add_argument("--temp-dir", type=Path)
    parser.add_argument("--save-features", type=Path, help="Write a new weighted global feature NPY; never overwrite")
    parser.add_argument("--joint-only", action="store_true", help="Suppress translation/rotation/hand diagnostic output")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error("Refusing to overwrite an existing result")
    try:
        report = evaluate_dataset(args.inputs, asset_root=args.asset_root, mode=args.mode,
            num_pairs=args.num_pairs, pairs_per_grasp=args.pairs_per_grasp, seed=args.seed,
            batch_size=args.batch_size, feature_batch_size=args.feature_batch_size,
            protocol=args.protocol, selection=args.selection, temp_dir=args.temp_dir,
            save_features=args.save_features, diagnostics=not args.joint_only,
            progress=None if args.quiet else lambda message: print(message, file=sys.stderr))
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.error(str(error))
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
