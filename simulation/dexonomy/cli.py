"""Small public interface for the V1.3.1 six-direction pipeline."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path


def check_assets(asset_root):
    root = Path(asset_root).resolve() / "hand" / "sharpa_v2"
    manifest = json.loads((Path(__file__).parent / "config" / "asset_manifest.json").read_text())
    errors = []
    for relative, expected in manifest["files_sha256"].items():
        path = root / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"hash mismatch: {relative}")
    actual = {p.stem for p in (root / "init_tmpl").glob("*.npy")}
    if actual != set(manifest["template_names"]):
        errors.append("template library must be exactly the accepted 35 families")
    if errors:
        raise ValueError("Asset validation failed:\n" + "\n".join(errors))
    return {"hand": "sharpa_v2", "template_count": len(actual), "files_checked": len(manifest["files_sha256"])}



def main(argv=None):
    p = argparse.ArgumentParser(description='Check externally supplied Sharpa production assets.')
    p.add_argument('--asset-root', required=True)
    args = p.parse_args(argv)
    print(json.dumps(check_assets(args.asset_root), indent=2))


if __name__ == '__main__':
    main()
