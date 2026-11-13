# Attribution and upstream license declaration

This code is derived from **Dexonomy: Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy**, by Jiayi Chen, Yubin Ke, Lin Peng and He Wang (RSS 2025).

- Upstream repository: https://github.com/JYChen18/Dexonomy
- Upstream base commit: `d561ff9df8545deef7a3f9e7cfb4045ef66f6913`
- Paper: https://arxiv.org/abs/2504.18829
- The upstream README declares the work and dataset licensed under **CC BY-NC 4.0**: https://creativecommons.org/licenses/by-nc/4.0/

This snapshot preserves that attribution and upstream license declaration. It does not grant rights to the excluded Sharpa models/templates, object assets, third-party libraries or datasets; those are supplied separately under their respective terms.

Local V1.3.1 modifications include mixed-object MuJoCo Warp physics, resident static model pools, shared-memory scheduling, online template feedback integration and scene input/output adapters. This review export removes unrelated execution paths and ends at six-direction force validation. See `SOURCES.json` for the source snapshot and extraction map.

```bibtex
@article{chen2025dexonomy,
  title={Dexonomy: Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy},
  author={Chen, Jiayi and Ke, Yubin and Peng, Lin and Wang, He},
  journal={Robotics: Science and Systems},
  year={2025}
}
```

The opt-in ADMM implementation additionally follows **GraspADMM: Improving
Dexterous Grasp Synthesis via ADMM Optimization**, Liangwang Ruan, Jiayi Chen,
He Wang and Baoquan Chen (2026), https://arxiv.org/abs/2603.13832.
`dexonomy/op/gen_grasp_admm.py` and `config/op/grasp_admm.yaml` are restored
verbatim from the upstream commit above. The MJWarp adapter is a local hybrid
CPU/GPU implementation, not an upstream GPU implementation or an assertion of
numerical equivalence. The upstream license declaration remains unchanged.

This integration adds explicit prediction frame/joint/control contracts, CPU
diagnostics and local bounded generation entry points. The fixed source map
records original hashes; modified files are identified separately. PyTorch3D
rotation utilities retain their BSD license in licenses/PyTorch3D-BSD.txt.
