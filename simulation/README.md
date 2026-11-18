# DexGrasp simulation runtime

The project defaults to the existing `/opt/isaacsim` image environment. From
its repository root:

```bash
./scripts/setup_image_environment.sh
./scripts/python.sh -m dexonomy.preflight
./scripts/python.sh -m dexonomy.prediction_eval --help
```

Training and CPU prediction evaluation share the image's NumPy 2 runtime.
The image profile is MuJoCo/MJWarp 3.12.0 and NVIDIA Warp 1.19.0+cu12.
The mixed-world adapter supports this profile with explicitly disabled noslip
(`noslip_iterations=0`); nonzero noslip is rejected without changing the request.
Real CUDA contact regression covers different mesh geometry, missing geom slots,
and singleton rebinding against native CPU stepping. This is not a full grasp
success benchmark. Snapshot backend identities record the actual engine version.
The legacy 3.17/1.16 profile remains recognized; the 3.3.3 compatibility protocol
remains restricted to its previously audited engines.
See [usage and contracts](docs/generation-evaluation.md).

External Sharpa XML/mesh/templates and object scene assets must be provided
under their own licenses. No asset download is implicit. This code retains
Dexonomy's CC BY-NC 4.0 terms; see NOTICE and licenses.

Historical 4090 constraints are retained for reproducibility. See
[hardware environments](environments/README.md) for the unvalidated Blackwell
candidate. Generic scheduling permits explicit continuous mode; the task-level
validation budget is not a permanent production time limit.
