"""Inspect sibling reference checkouts; optionally clone explicitly named sources."""
import argparse
import json
from pathlib import Path
import subprocess

CONFIG = Path(__file__).resolve().parents[1] / "configs/references.json"


def main():
    sources = json.loads(CONFIG.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clone", nargs="+", choices=[r["name"] for r in sources], default=[])
    args = parser.parse_args()
    report = []
    for source in sources:
        root = (CONFIG.parent / source["path"]).resolve()
        if source["name"] in args.clone and not root.exists():
            root.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "-c", "credential.interactive=false", "clone", source["url"], str(root)],
                           check=True, timeout=120)
        missing = [entry for entry in source["code_entries"] if not (root / entry).is_file()]
        report.append({"name": source["name"], "path": str(root), "exists": root.is_dir(),
                       "missing_code_entries": missing, "scope": source["scope"]})
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return int(any(r["missing_code_entries"] for r in report if r["name"] in args.clone))


if __name__ == "__main__":
    raise SystemExit(main())
