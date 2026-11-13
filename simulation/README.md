# DexGrasp simulation runtime

This separately installed runtime supports Sharpa prediction evaluation and
bounded local Floating / Tabletop generation. It does not install into the
training environment: its pinned NumPy 2 runtime conflicts with training's
NumPy <2 constraint. See [usage and contracts](../docs/generation-evaluation.md).

```bash
python3.11 -m venv .venv-simulation
.venv-simulation/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv-simulation/bin/pip install -c simulation/environments/rtx4090-tested.lock.txt -e ./simulation
.venv-simulation/bin/python -m dexonomy.prediction_eval --help
.venv-simulation/bin/python -m dexonomy.generation --help
```

External Sharpa XML/mesh/templates and object scene assets must be provided
under their own licenses. No asset download is implicit. This code retains
Dexonomy's CC BY-NC 4.0 terms; see NOTICE and licenses.

The commands above reproduce the tested 4090 environment. See
[hardware environments](environments/README.md) for the unvalidated Blackwell
candidate. Generic scheduling permits explicit continuous mode; the task-level
validation budget is not a permanent production time limit.
