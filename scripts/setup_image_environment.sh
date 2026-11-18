#!/usr/bin/env bash
# Add project dependencies while constraining every existing image package.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python="$repo/scripts/python.sh"
"$python" -c 'from importlib.metadata import version; assert version("isaacsim") == "6.2.0.0", "Expected Isaac Sim 6.2 image"'
snapshot="$(mktemp)"
trap 'rm -f "$snapshot"' EXIT
# Only the three project-owned helper packages may follow the reviewed profile.
# Pin installed versions, not build-time file:// wheelhouse URLs (often removed).
"$python" - <<'PYTHON' > "$snapshot"
from importlib.metadata import distributions
for dist in sorted(distributions(), key=lambda d: d.metadata['Name'].lower()):
    name = dist.metadata['Name']
    if name.lower().replace('_', '-') not in ('epic-dexgrasp', 'dexgrasp-simulation',
                                               'qpsolvers', 'hydra-core', 'ruamel.yaml', 'warp-lang'):
        print(f'{name}=={dist.version}')
PYTHON
"$python" -m pip install -c "$snapshot" wheel
# PyPI Warp 1.19 uses CUDA 13; the R570 image needs the official CUDA 12 build.
if ! "$python" -c 'from importlib.metadata import version; assert version("warp-lang") == "1.19.0+cu12"'; then
    "$python" -m pip install --no-deps 'https://github.com/NVIDIA/warp/releases/download/v1.19.0/warp_lang-1.19.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl#sha256=f95d9581ed12663df272e45a50101b171e143fde1ef7b2433a8bfc9fba096598'
fi
"$python" -m pip install --no-build-isolation -c "$snapshot" \
    -c "$repo/simulation/environments/isaacsim-6.2.txt" \
    -e "$repo" -e "$repo/simulation"
"$python" -m pip check
"$python" -m dexonomy.preflight
