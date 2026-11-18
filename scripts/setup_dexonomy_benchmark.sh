#!/usr/bin/env bash
# Install upstream CPU simulation in a project-private interpreter.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
profile="${1:-dexonomy-benchmark}"
case "$profile" in
    dexonomy-benchmark) venv="$repo/.venv-dexonomy-benchmark" ;;
    sharpa-production-replay) venv="$repo/.venv-weilun-source" ;;
    *) echo "Use dexonomy-benchmark or sharpa-production-replay" >&2; exit 2 ;;
esac
image_python="$repo/scripts/python.sh"
if [[ ! -x "$venv/bin/python" ]]; then
    "$image_python" -m venv --without-pip "$venv"
fi
"$image_python" -m pip --python "$venv/bin/python" install -r "$repo/simulation/environments/$profile.txt"
"$image_python" -m pip --python "$venv/bin/python" check
