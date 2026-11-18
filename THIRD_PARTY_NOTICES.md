# Third-party code

`dexgrasp/vendor/mink_unet.py` retains the sparse `ResNetBase`, `MinkUNetBase`
and `MinkUNet14` definitions from KPGrasp's
`src/dexgrasp/network/backbones/mink_unet.py`, inspected at commit
`89716dd5953ff3c88effbad7e7d4312c82653279` of
[KPGrasp](https://github.com/Yuansen-Huang/KPGrasp).
These definitions originate in NVIDIA MinkowskiEngine, Copyright NVIDIA
Corporation and Chris Choy, and retain their MIT license in
`licenses/MinkowskiEngine-MIT.txt`, as identified by KPGrasp's `NOTICE.md`.

The KPGrasp point-cloud wrappers and PointNet2 FPS helper were not copied.
The new wrapper uses a one-channel target mask, TensorField inverse mapping,
independent object/scene encoders and an explicit projection. The sparse
architecture itself is unchanged. FPS and all other v1 model/data/training/
geometry code here were implemented for this design.

The design uses the keypoint representation and geometric decoding idea from
[KPGrasp](https://arxiv.org/abs/2606.09314); this repository does not claim to
reproduce its published experiments or a complete Flow Map Matching objective.
The training package does not redistribute full Sharpa models or meshes.
The simulation diversity evaluator includes a minimal diagnostic keypoint
mapping; see the source and license details below.

`scripts/patches/minkowski-cuda12.patch` is a compatibility patch for the official
[NVIDIA/MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) source at
`02fc608bea4c0549b0a7b00ca1bf15dee4a0b228`. It adds explicit Thrust includes,
uses the installed NVTX C++ API when available, and preserves the hash map's
custom deleter while avoiding a GCC13/CUDA shared-pointer conversion ambiguity.
It does not change convolution equations. The patched source retains its
original headers: MinkowskiEngine MIT and the included NVIDIA/cuDF Apache-2.0
headers. The patch is applied only to the project-local dependency checkout.

## Source-backed Sharpa diagnostic dataset

`build_dexonomy_sharpa.py` consumes existing [Dexonomy data](https://huggingface.co/datasets/JiayiChenPKU/Dexonomy), licensed CC BY-NC 4.0, at revision `aa432b9f9d7a7355e9ab4fa881154a5f1ac4f881`. Data stays outside this repository. Each retained record traces to an archive member, source row, object mesh, scene configuration and per-grasp scale. The source pool includes upstream simulation-successful Shadow poses; this success label is **not** transferred to Sharpa.

The Shadow kinematics input is `assets/hand/shadow/right.xml` from [Dexonomy](https://github.com/JYChen18/Dexonomy), commit `d561ff9df8545deef7a3f9e7cfb4045ef66f6913`, based on the Apache-2.0 MuJoCo Menagerie Shadow model. The copied dataset asset retains the Shadow Apache-2.0 notice. The adapter is original project code and strips geometry for kinematics only. Sharpa asset attribution remains as documented above.

## Geometry and force-test evaluation

`dexgrasp/test/physics.py` and `scripts/evaluate_physics.py` independently implement
metric definitions described by KPGrasp, Dexonomy and DexGraspBench; no source
implementation is copied. The reference definitions are attributed in README.
The adapted single-pose Sharpa protocol is not the original Shadow benchmark.
The asset preparation command downloads official Sharpa XML and collision meshes
at revision `d323b371d02f4fc04a0873c19c5234cae6d803dd` into the outer `data/`
directory, together with the upstream LICENSE.txt and NOTICE.txt and file hashes.
The original XML is preserved; the derived XML removes noncollision visuals only.
CoACD is used as an installed dependency, not vendored source. Its generated
object collision meshes retain the archived source mesh identity in manifests.

## Muon compatibility

`dexgrasp/vendor/muon.py` is PyTorch v2.9.0 `torch/optim/_muon.py`,
with an absolute optimizer import and a local scalar helper for PyTorch 2.7.
Source: https://github.com/pytorch/pytorch/blob/v2.9.0/torch/optim/_muon.py
The full upstream copyright and license are in `licenses/PyTorch-LICENSE.txt`.

## Optional Dexonomy simulation runtime

`simulation/` is separately installed and derived from Dexonomy (Jiayi Chen,
Yubin Ke, Lin Peng and He Wang), upstream d561ff9 and integration 9c220bd.
See `simulation/NOTICE.md`, `simulation/SOURCES.json`, and the complete
`simulation/licenses/CC-BY-NC-4.0.txt`. The noncommercial upstream terms are
retained. `simulation/dexonomy/util/torch_util.py` also derives from PyTorch3D;
its BSD license is in `simulation/licenses/PyTorch3D-BSD.txt`.
Full hand models, annotation packages, templates and object assets are not
redistributed. The minimal diversity landmark mapping is the exception
documented below, with its source and license retained.

## Sharpa diversity landmark mapping

`simulation/dexonomy/diversity/resources/sharpa21_v1.json` contains the team's
21 diagnostic landmarks and the kinematic offsets needed to use them with
Dexonomy's Sharpa V2 model. It is extracted from `sharpa_21point_v1`, derived
from [Sharpa's URDF](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml) at
`d323b371d02f4fc04a0873c19c5234cae6d803dd`. Source file SHA256 values are retained
in the JSON. Copyright 2025 Sharpa Group; the original LICENSE and NOTICE
and full Apache-2.0 text are retained beside that resource. See also
[simulation attribution](simulation/NOTICE.md). This is a project-defined
kinematic representation, not a manufacturer-defined 21-point standard,
measured contact surface or hardware calibration. Full assets remain external.

Image build adaptation: `scripts/patches/minkowski-isaac-cu128.patch` applies to the
checksum-pinned NVIDIA MinkowskiEngine 0.5.4 source in `minkowski-source.json`.
It includes CUDA 12/CCCL/GCC compatibility and links the image scipy-openblas32
BLAS symbols. It is a complete alternative to `minkowski-cuda12.patch`, not an
additional patch to apply on top. NVIDIA MIT license: `licenses/MinkowskiEngine-LICENSE.txt`.
