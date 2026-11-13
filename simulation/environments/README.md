# Hardware environments

| Profile | Status | Torch wheel | Physics |
| --- | --- | --- | --- |
| `rtx4090-tested.lock.txt` | Bounded validation on RTX4090 / Linux x86_64 / Python 3.11 | 2.5.1+cu121 | MuJoCo/MJWarp 3.17.0, Warp 1.16.0 |
| `blackwell-candidate.txt` | RTX5090 **NOT_RUN** | 2.7.0, CUDA 12.8 | Same physics pins; no automatic upgrade |

PyTorch 2.7 introduces Blackwell support and CUDA 12.8 wheels, including updated
libraries and Triton. This is a dependency candidate, not evidence that this
application runs on RTX5090. [Official release](https://pytorch.org/blog/pytorch-2-7/).

Warp 1.16 PyPI wheels use CUDA 12.9; its fixed-tag documentation specifies driver
525+ for CUDA 12 builds and 580+ for CUDA 13 builds. Those generic floors do not
establish support for a new GPU: use a driver supporting the actual hardware and
all selected runtimes, then perform JIT/physics validation.
[Warp 1.16 installation](https://github.com/NVIDIA/warp/blob/v1.16.0/docs/user_guide/installation.rst).
CUDA 12.8 release notes list Linux 570.86.10 with the toolkit, separately from
minor-version compatibility floors.
[CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html).

Example fresh environments (do not mutate an active shared environment):

```bash
# Reproduce tested 4090 package versions.
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -c simulation/environments/rtx4090-tested.lock.txt -e ./simulation

# Candidate only; requires subsequent validation on actual 5090 hardware.
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -c simulation/environments/blackwell-candidate.txt -e ./simulation
python -m dexonomy.preflight
```

The metadata-only preflight initializes no GPU. The guarded runner performs the
GPU part after exclusive locks and PID ownership. It records visible GPU name,
compute capability, actual total memory, driver, Torch CUDA runtime/architectures,
and Warp device. `CUDA_VISIBLE_DEVICES` selects the allocated physical GPU;
`cuda:0` inside the worker means its first visible device, not physical GPU zero.
The preflight rejects unsupported Blackwell Torch/CUDA combinations and records
5090 acceptance as NOT_RUN until an actual validation report exists.

Configure CPU IDs from process affinity/cgroup limits. Configure Init, Refine,
Eval batches and contact capacities for each device **and asset set**; do not
scale a 4090 batch solely by advertised VRAM. The guard reads actual memory and
retains reserve thresholds. It never assumes 24 GB or a fixed CPU topology.
Cache paths are configuration-controlled; cold/JIT timings are not warm timings.
