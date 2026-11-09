# Reference：使用哪些代码

`../reference/` 是上游源码阅读与复用区。自有模型/训练改动在本仓库实现；不把五个仓库当作必须同时安装、同时执行的整体依赖。`configs/references.json` 是来源与接口索引，不是版本锁定文件。没有要求 checkout 到某个指定 commit。

实际运行实验时，自有代码仍记录配置和源码指纹；FK/IK 记录真正调用的源文件/资产指纹，便于定位结果。这与日常引用五个参考仓库的版本不同。已有上游 Git 历史保留，不删除、不合并进自有仓库。

## 数据链路

**Dexonomy** 提供抓型、模板和数据生成思路，入口是 `dexonomy/main.py`、`script.py`、`script_admm.py`，操作教程在 `getting_started/`。学习用数据从其发布数据集获取；生成代码依赖不等于训练依赖。BODex 子模块用于可选轨迹合成，不应为读取已有训练数据而安装整套生成环境。

**KPGrasp** 的 `docs/DATA.md` 和 `dataset/kdex.py` 定义关键点训练格式。`data_proc.py` 打包已有关键点，不负责原始 qpos 转关键点；`pc_proc.py` 打包点云。自有数据适配器读取相同字段，按行访问 HDF5，独立建立 val。`prepare.py` 从上游 IK 文件读取 28 个 site 的顺序，并使用上游 Shadow Hand XML 做 FK；没有导入 KPGrasp 的训练网络。

**DexLearn** 的 `dataset/base_dex.py` 读取三阶段 qpos 和局部点云，支持抓型条件。它与 KPGrasp 的完整点云/关键点输入不同；数据格式转换时必须核对坐标系、scale、关节顺序和 quaternion，不能仅改数据路径。

**DexGraspNet2** 的 `src/utils/dataset.py` 组织杂乱场景、graspness、抓取姿态和场景划分；默认灵巧手配置为 Leap Hand。这套场景数据不会通过 Shadow28 adapter 自动兼容。

## 模型与训练

| 参考仓库 | 重点代码 | 可复用/对照的内容 | 当前接入 |
| --- | --- | --- | --- |
| KPGrasp | `network/model.py`、`flow_method.py`、`policy/transformer.py`、`ode_solver.py` | 关键点 token、点云条件、FM/FlowMap、采样器 | 官方命令规划；自有训练并未使用其 MinkUNet/DiT |
| DexLearn | `network/models/base.py`、`type_emb.py`、`final_layers/diffusion.py`、`normalizing_flow.py` | 条件组合、扩散与 NF 输出头、分阶段姿态 | 官方命令规划；尚未移植头部代码 |
| DexGraspNet2 | `network/graspness_sample.py`、`diffusion.py`、`condition.py` | graspness 条件、位姿扩散、关节预测 | 官方命令规划；独立数据/手型协议 |

自有 `models.py`/`methods.py` 是统一接口下的轻量基线。阅读这些参考实现不等于声称实现了对应论文模型。今后真正移植某个模块时，应记录“来源仓库/文件、采用的接口、修改内容”，再增加它自己的配置和验证。

保留的核查事项：KPGrasp 默认 FM 与 FlowMap 深度不同；KPGrasp/DexLearn 上游训练验证使用 `test.json`；DexLearn 默认仅一种抓型。做公平对照时分别处理，不能只统一命令名字。

## 模型 test 链路

**KPGrasp IK**：`src/dexgrasp/kinematics/ik.py`、`assets/hand/shadow/right_hand_w_kp.xml`。自有 `test.py` 动态调用这个求解器并自行管理输出目录；没有调用上游会限制输出位置的 `ik.py` CLI。依赖 MuJoCo、mink、transforms3d、quadprog 等；当前求解器运行环境尚未完成验证。

**DexGraspBench**：`src/main.py` 分派 format/eval/stat；`task/convert_format.py` 的 Learning adapter 读取三阶段 qpos 和 scene_path；`task/evaluation.py`、`task/eval_func/` 做物理测试；`task/statistic.py` 汇总。手模型、接触、质量、摩擦和判据必须在同一对照中保持一致。自有代码当前只生成命令计划，不运行其含清理操作的 shell 脚本。

**DexGraspNet2**：`src/eval/` 有它自己的推理与 Isaac Gym 物理测试。这是独立路线，不要求自有监督 train 安装 Isaac Gym 或 Isaac Lab。

## 来源、第三方资产与准备方式

五个上游分别来自 [KPGrasp](https://github.com/Yuansen-Huang/KPGrasp)、[DexLearn](https://github.com/JYChen18/DexLearn)、[DexGraspNet2](https://github.com/PKU-EPIC/DexGraspNet2)、[Dexonomy](https://github.com/JYChen18/Dexonomy)、[DexGraspBench](https://github.com/JYChen18/DexGraspBench)。移植代码/资产时保留该部分实际适用的署名和许可证；不能把第三方内容自动改成自有代码许可证。

KPGrasp `NOTICE.md` 已分别列明 DiT、MinkowskiEngine、Muon、Shadow Hand 等来源。自有仓库当前没有复制这些网络实现或手模型；主要是读取/调用 sibling reference。上游各自的 `.gitmodules` 及子模块状态保持原样；缺少子模块不影响阅读源码，但会影响对应运行功能。

`python scripts/setup_references.py` 检查五个源码目录和关键文件；`--clone <名称...>` 只克隆缺失的指定仓库，不 pull、不重置本地修改、不安装环境、不下载数据和权重。实际使用哪个模块，再准备哪套依赖。
