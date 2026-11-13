# Bounded Sharpa integration validation

Validated implementation: `6774c1294f0f320f4e25c6570dc568e2642f87bb`.
Upstream extraction: `9c220bd8127eb0a1afd76e9c7ab5b48abf92d277`.
This report covers an isolated environment and small local workloads, not full-dataset
production acceptance. Raw data, local paths, checkpoint artifacts and logs remain
outside Git. See [interface and protocol](generation-evaluation.md) and
[hardware profiles](../simulation/environments/README.md).

## Software and environment

- DexGrasp regression: 49 tests run, 1 skipped, no failures; training tests keep
  `optimizer.step` mocked.
- Physics contract, portability and lightweight import: 12 tests passed.
- Production transaction, recovery and configuration: 45 tests passed.
- RTX4090, Python 3.11.14, Torch 2.5.1+cu121, MuJoCo/MJWarp 3.17.0,
  Warp 1.16.0, Linux driver 570.172.08; exact package lock is provided separately.
- All 13 guarded GPU attempts: 523.25 seconds total including startup and JIT;
  longest attempt 73.81 seconds. Three compatibility locks, PID challenges and
  resource guards remained enabled. No training or continuous production ran.
- RTX5090: **NOT_RUN**. Torch 2.7/CUDA 12.8 is a candidate environment, not an
  application validation. Physics versions are unchanged.

## Fixed real inputs

Four historical Sharpa refined inputs retain full pregrasp trajectories and squeeze
controls: one known success and failure per protocol. These are decoded-interface
fixtures derived from real grasp inputs, **not checkpoint predictions**. Named joint
mapping is exact, translation error is zero, and rotation quaternions are equivalent.
The exported JSON request path was also exercised on GPU.

| Protocol | Candidates | CPU success / failure / error | GPU success / failure / error |
| --- | ---: | --- | --- |
| Floating | 2 | 1 / 1 / 0 | 1 / 1 / 0 |
| Tabletop | 2 | 1 / 1 / 0 | 1 / 1 / 0 |

Floating records all six independent force directions for both candidates. Tabletop
success records six directions starting from the complete Slide terminal state;
the failing candidate fails Slide and therefore has no FC measurements. Collision-free
rate remains unknown; no legacy ray-gap joint metric is fabricated. Static mesh
previews were inspected separately and do not count as physical evidence.

Cross-process repeated GPU evaluations preserve all 4 candidate labels. Maximum
absolute differences among corresponding reported translation/rotation metrics:

| Protocol | Translation | Rotation |
| --- | ---: | ---: |
| Floating | 0.468 mm | 1.0733 degrees |
| Tabletop | 0.0508 mm | 4.5903 degrees |

These include failing trajectories and do not establish bitwise reproducibility.
CPU/GPU label agreement on four cases does not establish physical equivalence.

## Generation and recovery

Final frozen-source runs invoke `dexonomy.generation run` with one object, all six
scales, finite sampling, sequential scheduling and per-stage batch capacities.
Floating uses one grasp family; Tabletop uses two. Batch settings are workload
choices, not universal GPU defaults.

| Protocol | Init accepted | Refined | Slide passed | Final success | Feedback | FC directions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Floating | 6 | 4 | n/a | 2 | 2 | 24 |
| Tabletop | 57 | 9 | 8 | 5 | 5 | 48 |

Independent artifact readback verifies stage hashes, feedback archives, all recorded
pose thresholds, and success composition. Tabletop additionally verifies eight
complete terminal snapshot hashes and all six direction start hashes per passing
Slide; table recontact participates in each direction's pass decision. Successful
outputs retain terminal state, controller target and actual object pose.

Both final checkpoints were resumed through `dexonomy.generation resume`.
Completed jobs, stage rows, artifact hashes and feedback IDs are unchanged: no
repeat generation or duplicate feedback. First-run quotas were not universally
filled; bounded shortfalls are retained rather than reported as completed targets.
An earlier Tabletop attempt produced 6 Init and zero Refine, hence no physical Eval;
that attempt is retained alongside the later complete chain.

The final guarded production attempts took 43.5 s (Floating) and 58.5 s (Tabletop),
including runner startup. These are small warm-cache end-to-end observations,
not throughput benchmarks; the first fixed-input floating cold/JIT attempt took
73.8 s with a different workload. They must not be compared as speedup figures.

## Remaining boundaries

The target main's three core training/inference/decode source hashes do not match
the supplied checkpoint provenance. Checkpoint inference and model quality are
**NOT_RUN** until matching source and hand specification are provided. The independent
physics interface and generation implementation do not depend on resolving that mismatch.

Large datasets, long continuous runs, multi-GPU/cloud production, other hand models,
and 5090 runtime/physics acceptance remain unvalidated. The product accepts explicit
continuous configuration, while this validation uses only the authorized bounded budget.
Failed setup/download attempts and intermediate mapping corrections remain in the
local evidence trail; they are not counted as physical successes.
