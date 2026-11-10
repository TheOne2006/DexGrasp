# 项目结构与 Git 边界

外层是研究工作区，内层同名目录是自有 Git/GitHub 仓库。外层资料、数据与参考仓库保持原位；本次重建的是内层自有代码。

| 内层目录 | 职责 |
| --- | --- |
| `dexgrasp/data/` | 原始场景制作、Data v1 HDF5、DataLoader、数据划分审计 |
| `dexgrasp/models/` | ObjectEncoder、SceneEncoder、条件 Transformer、FM/FlowMap 参数化 |
| `dexgrasp/geometry/` | Sharpa 手模型协议、FK、掌面间距、SVD 和 IK |
| `dexgrasp/train/` | 速度回归、教师轨迹蒸馏、训练/验证/恢复 |
| `dexgrasp/test/` | 采样、解码、条件指标与外部碰撞/物理结果汇总 |
| `dexgrasp/vendor/` | 只收纳实际复用并保留来源的第三方代码 |
| `configs/` | 自有 FM/FlowMap 配置、标定模板与参考源码索引 |
| `tests/` | 软件单元和集成测试；不存真实实验数据 |
| `scripts/` | 参考检查、软件性能检查等维护工具 |
| `docs/` | 随代码发布的接口、架构和验证说明 |
| `licenses/` | 被采用模块的第三方许可证 |
| `.github/` | CI 和 PR 模板 |

Python 包统一为 `dexgrasp`，命令入口 `python -m dexgrasp`。旧 `dexgrasp_lab`、Shadow 三阶段基线及兼容命令已移除，不与 v1 共用数据协议。

外层目录职责：`../data/` 保存原始和处理后数据；`../checkpoints/` 保存模型/优化器状态；`../outputs/` 保存日志、采样、解码与验证报告；`../reference/` 保留独立上游 Git 历史；`../docs/` 保留阅读路线与研究结论。论文库由 Zotero 管理。

配置中的 `../../data` 路径以 `configs/*.json` 所在目录为基准。正式运行先冻结代码、配置、划分和评价协议，记录实际使用的手模型/教师指纹。

Git 操作只在内层执行。保留原有 `origin` 配置及仓库可见性；本次不提交、不推送。参考源码不作为本仓库 submodule，不要求同时安装或运行所有参考项目。参考目录导航由 `configs/references.json` 和 `scripts/setup_references.py` 维护，具体采用模块见 `THIRD_PARTY_NOTICES.md`。
