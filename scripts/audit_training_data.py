"""Complete deep data/FK/leakage checks before launching distributed training."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dexgrasp.data.audit_receipt import create_receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--coord-scale-m', type=float, default=.2)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    receipt = create_receipt(args.dataset_root, args.coord_scale_m)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps(receipt['identity']['samples']))


if __name__ == '__main__':
    main()
