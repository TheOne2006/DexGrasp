#!/usr/bin/env bash
# One interpreter for project scripts, multiprocessing and installed entrypoints.
set -euo pipefail
runtime="${DEXGRASP_PYTHON:-/opt/isaacsim/bin/python}"
if [[ ! -x "$runtime" ]]; then
    echo "Missing project interpreter: $runtime" >&2
    exit 1
fi
# WebIDE's old SQLite shadows Ubuntu's Python 3.12-compatible library.
# Filter only this injected directory; retain CUDA/NVIDIA and user paths.
IFS=: read -r -a library_paths <<< "${LD_LIBRARY_PATH:-}"
clean_paths=()
for path in "${library_paths[@]}"; do
    [[ "$path" == /kaic/webide/lib ]] || clean_paths+=("$path")
done
export LD_LIBRARY_PATH="$(IFS=:; echo "${clean_paths[*]}")"
exec "$runtime" "$@"
