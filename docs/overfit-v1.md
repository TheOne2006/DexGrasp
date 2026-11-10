# Sharpa 小数据过拟合实验

日期：2026-11-10（Asia/Shanghai）。本轮获授权运行小规模优化器训练；不提交或推送代码。

## 数据与验收目标

数据位于外层 `data/overfit_sharpa_kinematic_v1/`。6 个不同长宽高比例的程序生成长方体、每个物体 2 个姿态，总计 12 条：train 8 / val 2 / test 2。不同物体独立划分，每个观测的两路点云只存一次。点云为目标独立 1024、场景重新采样目标 512＋桌面 1536 点，完整 mesh 表面积均匀采样，单位米。

这是 **synthetic kinematic diagnostic**，不是成功抓取数据集。使用 [Sharpa 官方模型](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml)，revision `d323b371d02f4fc04a0873c19c5234cae6d803dd`。官方 URDF、Apache-2.0 LICENSE/NOTICE 和来源哈希保存在 `data/raw/sharpa_wave/`，数据集内保留副本。原始 URDF 不改写；另建固定根坐标变换的派生 URDF，使项目 +z 对应官方根坐标 +x。

本轮 21 点采用 wrist＋每指 4 个 URDF 定义位置。关节中心使用父 link 的 joint origin，指尖使用官方 fingertip link 原点。该映射不是经过确认的 KPGrasp Figure 9 标定。`hand_spec.json` 的 calibrated 表示这些数值已明确且通过 FK 检查，其 `calibration_scope` 明确限定为诊断，不表示硬件或论文标定完成。

掌面测量区是显式指定的实验平面，64 个规则网格射线点，不宣称是真机掌面皮肤测量。每条姿态保存 R/t/q，K 由官方 22 关节 FK 得到，d=R[:,2]，h 由完整目标 mesh 首次命中距离的中位数得到，全部 64 射线命中。`reference_length_m` 由零位 wrist 到 middle-tip 的距离计算。所有关节在 URDF 限位内；没有接触、碰撞或抓稳验证，原始记录不伪造 valid=true 标志。

本地 Dexonomy 预览为 Shadow 手，DexGraspNet2 预览为 Leap 手；未跨手型直接复制关节。公开 [SharpaDex](https://huggingface.co/datasets/Sharpa-Robotics/SharpaDex-v1.0) 为遥操作数据，其公开特征表未提供本任务所需的配套目标 mesh 位姿标签；本轮不将其直接充当 Data v1 抓姿数据。

实验前已固定的目标：完整模型在 train 条件上，32 组独立新噪声的平均关键点欧氏误差 MPJPE < 5 mm。全部候选计入，不做 best-of 筛选。val/test 物体不用于选权重；本轮明确以训练集重建误差选 best，不能把这个选择规则用于正式泛化实验。

## 初始化与冻结

| 阶段 | 初始化 | 更新 | 冻结 |
|---|---|---|---|
| FM | seed=42；PyTorch/MinkowskiEngine 各模块默认随机初始化，不加载其他模型权重 | 两个独立 MinkUNet14、FPS 后投影/位置、8 层 Transformer、keypoint ID、方向 bias、h/time MLP、输出头；共 91,662,659 参数 | 无；编码器 BatchNorm 统计也更新 |
| FlowMap | 复制选定 FM 的兼容权重；时间层前 128 列复制 FM，新增后 128 列置零；学生独立存储 | 全部学生参数，包括两路编码器与 BN | 整个 FM 教师，包括 BN 统计；教师保持 eval/no_grad |

保留 width=512、depth=8、heads=8、tokens=128、MinkUNet 输出 512；未缩小成替代网络。FM 用 AdamW，lr=1e-4、weight_decay=0、clip=1、batch=8、FP32；每步重新采噪声与连续时间。初始预算 800 updates。FlowMap 配置 lr=5e-5、400 updates、训练条件下 Heun32 教师轨迹、50% 完整区间；推理一步。配置为 `configs/overfit_fm.json`、`configs/overfit_flowmap.json`。

通用配置另外支持 `freeze_encoders=true`，同时冻结编码器权重和 BN 运行统计；本轮为 false。没有可用匹配预训练权重时，不把随机编码器冻结作为默认方案。

## 存储与恢复

所有大文件位于外层：

```text
data/overfit_sharpa_kinematic_v1/           # assets / scenes / raw_grasps / hand / shards / splits
checkpoints/overfit_sharpa_v1/fm/          # best.pt / last.pt / resume_step0100.pt
checkpoints/overfit_sharpa_v1/fm_refine/   # 较低学习率的继续拟合，独立 run
checkpoints/overfit_sharpa_v1/flowmap/     # 学生权重
checkpoints/overfit_sharpa_v1/flowmap_euler20/ # 最终推荐的一步学生权重
outputs/overfit_sharpa_v1/                # protocol.json、源码快照、日志、评估结果
```

`best.pt` 按固定诊断噪声 seed=20042 下 train MPJPE 选取；`last.pt` 保存最近更新。checkpoint 包含模型、AdamW、更新步数、torch CPU/CUDA RNG、配置、数据/划分/手模型/源码/原生扩展指纹、教师 SHA256；原子临时文件替换。数据身份包括 shard 内容 SHA256，不能只凭路径或文件大小恢复。

先在 100 步受控退出，保留 `resume_step0100.pt`，然后同配置继续。`--stop-after` 是本次执行边界，不修改持久配置。恢复要求配置、代码、数据、运行时一致；若更改训练预算或代码，应建立明确的新实验，不悄悄复用旧运行。

运行命令（内层仓库、项目 `.venv`，不修改共享环境）：

```bash
.venv/bin/python scripts/build_overfit_dataset.py \
  --source ../data/raw/sharpa_wave --root ../data/overfit_sharpa_kinematic_v1

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -m dexgrasp train --config configs/overfit_fm.json --execute --stop-after 100
cp ../checkpoints/overfit_sharpa_v1/fm/last.pt ../checkpoints/overfit_sharpa_v1/fm/resume_step0100.pt
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -m dexgrasp train --config configs/overfit_fm.json --execute \
  --resume ../checkpoints/overfit_sharpa_v1/fm/resume_step0100.pt

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python scripts/evaluate_overfit.py \
  --checkpoint ../checkpoints/overfit_sharpa_v1/fm/best.pt \
  --output ../outputs/overfit_sharpa_v1/fm-evaluation.json
```

构建器、训练和评估均拒绝无意覆盖已有结果；复现应更换根目录与 run ID。最终评估 seed=40000…40031，区别于训练 checkpoint 选择噪声。保存生成坐标 NPZ、各条件误差和 checkpoint SHA256。物理成功率保持 null。

## 验证与结果

新增初始化复制/独立存储、时间层映射、冻结编码器模式、checkpoint 初始化来源与身份保护测试。40 项软件测试通过；软件训练循环继续禁用 optimizer.step。真实实验另外验证 optimizer 状态与恢复后的权重变化，证据在 `outputs/overfit_sharpa_v1/resume_verification.json`；这不等同于真实 CUDA 完整作业逐位一致性证明。

首轮 FM800 在 32 组新噪声下 train MPJPE 为 10.16 mm，未达到预先定义的 5 mm 门槛。由此建立独立的 `fm_refine`：`initialization=checkpoint`，复制首轮 best 的模型权重，使用全新的 AdamW（lr=3e-5），预算 2400 updates，保持全部参数可训练。`configs/overfit_fm_refine.json`、`refinement_protocol.json` 和独立源码快照记录此决定，未改写首轮实验。FlowMap 的 teacher 指向 refinement 的 best。

首轮诊断评估已经展示过 val/test 误差，所以后续这些数值只作描述性报告，不称为盲测或泛化验收；继续拟合的决策只依据 train 的 5 mm 门槛。程序生成的微型数据及其特定条件本身也不足以评估真实泛化。

实验最终数值以 `outputs/overfit_sharpa_v1/*evaluation.json` 为准；不把 train 重建、IK 可达或合成几何结果写成物理抓取成功率。

## 最终结果与推荐权重

所有误差均为米制还原后的平均关键点欧氏距离（MPJPE）；每个 train 条件生成 32 次，共 256 个输出，seed=40000…40031，全部计入。验收门槛是整体平均 < 5 mm，不是每个关键点或每个条件都 < 5 mm。

| 实验 | 本轮执行 updates | 选中本轮 step | 推理 | train MPJPE | train 关键点 P95 | 验收 |
|---|---:|---:|---|---:|---:|---|
| FM 随机初始化 | 800 | 800 | Euler20 | 10.16 mm | 18.82 mm | 未通过 |
| FM 低学习率继续拟合 | 2400 | 1500 | Euler20 | **3.40 mm** | **6.40 mm** | 通过 |
| FlowMap，Heun32 教师轨迹 | 400 | 300 | 一步 | 5.42 mm | 见 JSON | 未通过 |
| FlowMap，改用 Euler20 教师轨迹 | 400 | 200 | 一步 | **4.69 mm** | **8.51 mm** | 通过 |

FM 最佳权重来自首轮 step800＋继续拟合 step1500；FlowMap 最佳权重来自 Heun32 学生 step300＋Euler20 学生 step200。表中的 step 是各独立 run 的本地计数，不是所有尝试的累计优化次数。两个最终权重分别为：

- `../checkpoints/overfit_sharpa_v1/fm_refine/best.pt`，SHA256 `b81a3b79659ee4d875195eb06cca337fc6e02e8093e42280178864676257b29b`。
- `../checkpoints/overfit_sharpa_v1/flowmap_euler20/best.pt`，SHA256 `fcc2bcbeb1e70ef60abc5a22a53bc13ab88874623d4ef24d47a40d6410e69feb`。

最终 FlowMap 的配置为 `configs/overfit_flowmap_euler20.json`：从已训练的 Heun32 学生 best 开始，重置 AdamW，lr=3e-5、400 updates；同一 FM 教师仍全部冻结，学生仍全部训练。首次 FlowMap 初始化保持前文的 FM 权重复制规则。

更换教师积分协议是本轮的实测发现：同一 FM 权重使用 Heun32，在 8 组噪声上得到 5.98 mm 的训练集终点误差，劣于其选权重时使用的 Euler20。Euler20 在 32 组噪声上为 3.40 mm，因此将蒸馏轨迹与已验收的教师推理协议对齐。保留原 Heun32 轮次和全部日志，未修改求解器实现，也不由此推断 Heun 一般不如 Euler。正式数据上应分别验收教师的积分协议。

最终学生与同噪声 Euler20 教师的终点差异为 3.51 mm；教师文件哈希在蒸馏后保持一致。最终 FM/FlowMap 的最大单条件平均误差分别为 4.30/5.17 mm。未见物体的描述性 val 误差为 53.00/52.45 mm，test 为 74.22/73.76 mm，说明这个微型长方体数据上的拟合不代表泛化成功；所有物体属于同一程序生成形状族。

单张 RTX 5090、batch=8、FP32、包含两路点云编码和生成，3 次热身＋7 次计时的中位数：FM20 198.3 ms/batch，FlowMap1 53.3 ms/batch，实测约 3.72×。这不包含模型加载、IK、碰撞、仿真或完整规划，不应称为抓取流水线提速。

最终评估与复现：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -m dexgrasp train --config configs/overfit_fm_refine.json --execute
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -m dexgrasp train --config configs/overfit_flowmap.json --execute
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -m dexgrasp train --config configs/overfit_flowmap_euler20.json --execute

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python scripts/evaluate_overfit.py \
  --checkpoint ../checkpoints/overfit_sharpa_v1/flowmap_euler20/best.pt \
  --output ../outputs/overfit_sharpa_v1/flowmap-euler20-evaluation.json
```

已有运行目录时应复制配置并更换 run/checkpoint 目录，依照父 checkpoint 顺序执行；上述命令说明依赖关系，不能覆盖本轮已完成记录。

检查证据：`outputs/overfit-software-tests-v2.log`（40 项通过）、`label_recheck.json`（全部 12 标签重算）、`resume_verification.json`（真实优化器恢复）、`distillation-check.json`（教师/学生误差与计时）。四个 `code_before_*.tar.gz` 保存对应阶段源码、配置、测试和许可证；运行配置及源码/数据/原生扩展指纹也存入各 run 的 provenance 和 checkpoint。构建脚本可重新生成数据；正式 Sharpa 标定、真实成功抓姿、IK 和物理抓取验证仍未完成。

![训练曲线](../../outputs/overfit_sharpa_v1/training-curves.png)

![首条训练条件与第一组新噪声的关键点重建](../../outputs/overfit_sharpa_v1/keypoint-reconstruction.png)
