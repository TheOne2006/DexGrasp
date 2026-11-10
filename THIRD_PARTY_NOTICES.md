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
No Sharpa assets or calibrated numeric keypoint mappings are redistributed.

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

The explorer bundles a subset of Noto Sans SC, Copyright The Noto Project Authors, under the SIL Open Font License 1.1; license at `site/public/fonts/OFL.txt`. React, Vite, Three.js and Playwright retain their package licenses in their respective dependency distributions.
