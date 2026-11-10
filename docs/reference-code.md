# Reference：使用哪些代码

`../reference/` 是上游源码阅读与复用区。自有模型/训练改动在本仓库实现；不把参考仓库当作必须同时安装、同时执行的整体依赖。`configs/references.json` 是来源与接口索引，不是版本锁定文件。没有要求 checkout 到某个指定 commit。

实际运行实验时，自有代码仍记录配置和源码指纹；FK/IK 记录真正调用的源文件/资产指纹，便于定位结果。这与日常引用这些参考仓库的版本不同。已有上游 Git 历史保留，不删除、不合并进自有仓库。

## 数据链路

**Dexonomy** 提供抓型、模板和数据生成思路，入口是 `dexonomy/main.py`、`script.py`、`script_admm.py`，操作教程在 `getting_started/`。学习用数据从其发布数据集获取；生成代码依赖不等于训练依赖。BODex 子模块用于可选轨迹合成，不应为读取已有训练数据而安装整套生成环境。

**KPGrasp** 的 `docs/DATA.md` 和 `dataset/kdex.py` 定义关键点训练格式。`data_proc.py` 打包已有关键点，不负责原始 qpos 转关键点；`pc_proc.py` 打包点云。当前自有数据接口已重建为 Sharpa 21 点 Data v1，不再读取上游 Shadow28 标签。FK 使用明确标定的 Sharpa URDF 与 link/局部坐标映射。

**DexLearn** 的 `dataset/base_dex.py` 读取三阶段 qpos 和局部点云，支持抓型条件。它与 KPGrasp 的完整点云/关键点输入不同；数据格式转换时必须核对坐标系、scale、关节顺序和 quaternion，不能仅改数据路径。

**DexGraspNet2** 的 `src/utils/dataset.py` 组织杂乱场景、graspness、抓取姿态和场景划分；默认灵巧手配置为 Leap Hand。这套场景数据不能直接当作 Sharpa Data v1，需要手型适配与重新验证。

## 模型与训练

| 参考仓库 | 重点代码 | 可复用/对照的内容 | 当前接入 |
| --- | --- | --- | --- |
| KPGrasp | `network/model.py`、`flow_method.py`、`policy/transformer.py`、`ode_solver.py` | 关键点 token、点云条件、FM/FlowMap、采样器 | 采用其源自 MinkowskiEngine 的 MinkUNet14 定义；条件 Transformer 和训练目标按自有 v1 实现 |
| DexLearn | `network/models/base.py`、`type_emb.py`、`final_layers/diffusion.py`、`normalizing_flow.py` | 条件组合、扩散与 NF 输出头、分阶段姿态 | 仅保留源码阅读索引，未接入自有模型 |
| DexGraspNet2 | `network/graspness_sample.py`、`diffusion.py`、`condition.py` | graspness 条件、位姿扩散、关节预测 | 仅保留源码阅读索引，独立数据/手型协议 |

自有代码现为 `dexgrasp/models/` 与 `dexgrasp/train/`，只实现用户的 FM 教师和 FlowMap 蒸馏学生，不保留旧基线入口。实际复用的稀疏网络定义及修改边界见 `THIRD_PARTY_NOTICES.md`；其余参考内容不代表论文复现。

保留的核查事项：KPGrasp 默认 FM 与 FlowMap 深度不同；KPGrasp/DexLearn 上游训练验证使用 `test.json`；DexLearn 默认仅一种抓型。做公平对照时分别处理，不能只统一命令名字。

## 模型 test 链路

**KPGrasp IK**：`src/dexgrasp/kinematics/ik.py`、`assets/hand/shadow/right_hand_w_kp.xml`。这是上游 Shadow Hand 的求解器，当前自有模型不调用它。自有 `geometry/` 使用标定 Sharpa URDF、SVD 根配准与 SciPy 有界 IK，并提供解析 FK Jacobian。

**DexGraspBench**：`src/main.py` 分派 format/eval/stat；`task/convert_format.py` 的 Learning adapter 读取三阶段 qpos 和 scene_path；`task/evaluation.py`、`task/eval_func/` 做物理测试；`task/statistic.py` 汇总。手模型、接触、质量、摩擦和判据必须在同一对照中保持一致。当前自有代码接收外部评估器逐候选的碰撞/物理 verdict；尚未接通 Sharpa 物理模拟协议。

**DexGraspNet2**：`src/eval/` 有它自己的推理与 Isaac Gym 物理测试。这是独立路线，不要求自有监督 train 安装 Isaac Gym 或 Isaac Lab。

## 来源、第三方资产与准备方式

原有五个上游分别来自 [KPGrasp](https://github.com/Yuansen-Huang/KPGrasp)、[DexLearn](https://github.com/JYChen18/DexLearn)、[DexGraspNet2](https://github.com/PKU-EPIC/DexGraspNet2)、[Dexonomy](https://github.com/JYChen18/Dexonomy)、[DexGraspBench](https://github.com/JYChen18/DexGraspBench)。移植代码/资产时保留该部分实际适用的署名和许可证；不能把第三方内容自动改成自有代码许可证。

KPGrasp `NOTICE.md` 已分别列明 DiT、MinkowskiEngine、Muon、Shadow Hand 等来源。自有仓库仅采用实际需要的 MinkUNet 稀疏网络定义，并保留其 MIT 许可证；没有复制 DiT、Muon 或 Shadow Hand 资产。上游各自的 `.gitmodules` 及子模块状态保持原样；缺少子模块不影响阅读源码，但会影响对应运行功能。

`python scripts/setup_references.py` 检查索引中的源码目录和关键文件；`--clone <名称...>` 只克隆缺失的指定仓库，不 pull、不重置本地修改、不安装环境、不下载数据和权重。实际使用哪个模块，再准备哪套依赖。

## 方向可控抓取参考（2026-11-09 加入）

以下四个独立仓库保留上游 Git 历史与原始许可证/署名，仅加入源码和接口索引；尚未安装依赖或运行论文基线。

| 仓库与官方来源 | 代码入口（相对于各仓库） | 阅读用途 |
| --- | --- | --- |
| [CGrasp](https://github.com/gpaschalidis/CGrasp) | `cgrasp/models/models.py` 的 `decode`、`cgrasp/train/trainer.py`、`generate_grasps.py` | MANO 人手方向向量条件，解码器将 `dir_hand` 与 BPS 拼接 |
| [CWGrasp](https://github.com/gpaschalidis/CWGrasp) | `optimization.py`、`cfg/loss_config.yaml` | CGrasp 所属的全身抓取框架与优化；CGrasp 为单独发布模块，因此两者均保留 |
| [CAPGrasp](https://github.com/wengzehang/CAPGrasp) | `grasp_sampler_refiner.py` 的 `forward`、`utils/utils.py`、`models/networks.py` | 接近方向坐标对齐、方向锥半角条件与 refinement；对象为平行夹爪 |
| [TRO-Grasp](https://github.com/Barrybarry-Smith/TRO-Grasp) | `model/tro_graph.py`、`model/denoiser.py`、`config/test_palm_conditioned.yaml` | 图扩散与 `palm_conditioned` 采样分支、旋转测地距离引导；约束完整掌部旋转 |

来源由论文链接的官方项目页确认：[CWGrasp](https://gpaschalidis.github.io/cwgrasp/)、[CAPGrasp](https://wengzehang.github.io/CAPGrasp/)、[T(R,O) Grasp](https://nus-lins-lab.github.io/trograspweb/)。CGrasp 的独立仓库由 CWGrasp 官方 README 指向。

**DCGrasp 暂缺代码**：截至 2026-11-09，[官方项目页](https://4dqv.mpi-inf.mpg.de/DCGrasp/)没有代码下载链接，GitHub 仓库名检索也未找到对应仓库。保留此缺项说明，暂不添加虚构的 clone 地址或空源码目录。论文中的方向与 root distance 条件不能代替已核验的实现；root distance 也不能直接视为本项目 palm_gap。

完整 CWGrasp 运行另需 CReach、ReachingField、MANO/SMPL-X、预训练模型和数据，当前未额外准备这些组件。这些参考不提供自有 baseline 启动入口；实际复现实验前需独立配置环境、数据与评价协议。上游许可证及第三方资产授权按原仓库处理。
