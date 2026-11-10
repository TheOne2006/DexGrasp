# DexGrasp explorer

Read-only research viewer for `../../data/sharpa_dexonomy_1k_v1` and `../../outputs/sharpa_dexonomy_1k`. All plots and geometry come from recorded data. The image-generated design concept is not served as scientific evidence.

Build with Node 22 and the lockfile: `npm ci && npm run build`. This host has Node but no global npm; a local npm bootstrap exists under ignored `.tools/package/bin/npm-cli.js`.

From the research workspace run:

```bash
DexGrasp/.venv/bin/python DexGrasp/scripts/serve_explorer.py --port 8766
```

The server binds only `127.0.0.1`. This host's existing Tailscale userspace networking forwards inbound traffic to the same local port. URL: `http://100.91.234.79:8766`. No Tailscale Serve/Funnel settings or system permissions were changed. The running server is recorded under `outputs/sharpa_dexonomy_1k/site_server.json`; it persists beyond the command session but is not a boot service.

The API permits summary, records, protocol, run metrics and sample IDs found in the finalized manifest. It does not expose checkpoints, arbitrary filesystem paths or the outer workspace. Browser dependencies were extracted locally under `.tools/browser-libs/`; no system packages were installed. Playwright evidence lives in `outputs/sharpa_dexonomy_1k/design/`.

Prediction overlays are stored per run: `outputs/sharpa_dexonomy_1k/{fm,flowmap}/site_predictions.npz`. Each contains all 1000 first-seed predictions (seed 40000), IDs and the evaluated checkpoint hash. `/api/sample/<id>` returns a `predictions` map keyed by `fm` and `flowmap`. The viewer defaults to FlowMatching (blue); FlowMap (purple) can be toggled independently, including simultaneous comparison. The old root-level `site_predictions.npz` is a historical artifact and is no longer read.

To rebuild an overlay from an existing evaluation, from the research workspace:

```bash
DexGrasp/.venv/bin/python DexGrasp/scripts/export_explorer_predictions.py --root data/sharpa_dexonomy_1k_v1 --evaluation outputs/sharpa_dexonomy_1k/fm/evaluation.json --output outputs/sharpa_dexonomy_1k/fm/site_predictions.npz
DexGrasp/.venv/bin/python DexGrasp/scripts/export_explorer_predictions.py --root data/sharpa_dexonomy_1k_v1 --evaluation outputs/sharpa_dexonomy_1k/flowmap/evaluation.json --output outputs/sharpa_dexonomy_1k/flowmap/site_predictions.npz
```
