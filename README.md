# DexGrasp

EPIC 的场景条件灵巧手抓姿生成框架，按个人模型设计 v1 实现。当前只包含自有 Sharpa 21 点模型：**Flow Matching 教师 + FlowMap 轨迹蒸馏学生**。

输入目标点云、包含目标/环境/桌面的场景点云、目标标记、连续 `palm_gap` 和必需的接近方向；输出整手关键点，再经掌部 SVD 与有界手指 IK 恢复 `R,t,q[22]`。旧 Shadow 三阶段基线、Diffusion 模型及其入口已移除。

## 文件夹职责

```text
DexGrasp/                         自有 Git 仓库
├── dexgrasp/
│   ├── data/                     Data v1：表面采样、HDF5、划分与数据审计
│   ├── models/                   两路 MinkUNet + 条件 Transformer + FM/FlowMap 输出
│   ├── geometry/                 手模型标定接口、FK/Jacobian、掌面测量、SVD/IK
│   ├── train/                    FM/蒸馏目标、训练、验证、检查点和断点恢复
│   ├── test/                     生成、几何解码、条件误差和外部物理结果汇总
│   └── vendor/                   保留许可证的 MinkUNet 稀疏网络定义
├── configs/                      FM、FlowMap、八卡训练、手模型模板与参考索引
├── tests/                        软件测试；合成夹具不用于研究结论
├── scripts/                      数据制作、关键点评估、八卡训练与软件检查
├── docs/                         model v1 架构（含原始输入）与 data 说明
├── licenses/                     第三方许可证
└── .github/                      CI
```

实际数据、权重和结果仍分别放在外层 `../data/`、`../checkpoints/`、`../outputs/`；参考源码留在 `../reference/`。数据展示板及其服务、展示导出工具位于同级的 [`../site/`](../site/README.md)，独立于本实验代码仓库。模型架构见 [model v1](docs/model-v1.md)，数据协议与来源见 [data](docs/data.md)。

## 运行

使用包含 PyTorch、NumPy、h5py、SciPy 的项目环境。真实几何编码还需要与 PyTorch/CUDA 匹配的 **MinkowskiEngine**；缺失时明确报错，不替换成其他模型。OBJ/STL/PLY 导入额外需要 `trimesh`；NPZ 三角网格不需要它。

```bash
# 在内层仓库中执行；先检查真实数据、手模型标定与依赖。
python -m dexgrasp plan --config configs/fm.json
python -m dexgrasp plan --config configs/flowmap.json

# 读取已准备好的原始 assets、scenes、hand 并生成 HDF5 与 splits。
python -m dexgrasp prepare-data --root ../data/dataset_v1
python -m dexgrasp audit-data --root ../data/dataset_v1

# 不带 --execute 只显示计划。FM 完成后再使用其检查点训练 FlowMap。
python -m dexgrasp train --config configs/fm.json --execute
python -m dexgrasp train --config configs/flowmap.json --execute

python -m dexgrasp sample --checkpoint ../checkpoints/v1/flowmap/best.pt \
  --dataset-root ../data/dataset_v1 --output ../outputs/v1/flowmap_samples.h5 \
  --steps 1 --device cuda:0
python -m dexgrasp decode --samples ../outputs/v1/flowmap_samples.h5 \
  --output ../outputs/v1/decoded.jsonl
python -m dexgrasp test --decoded ../outputs/v1/decoded.jsonl
```

配置内路径相对于配置文件，URDF 路径相对于 `hand_spec.json`。恢复使用同一配置加 `--resume /absolute/path/last.pt`。运行前写入源码、配置、划分、手模型和教师指纹；拒绝覆盖已有运行目录。

数值修复续训可通过 DDP 的 `--fork-from /absolute/path/step_XXXXXX.pt` 创建独立运行：
保留权重、优化器矩、逐 rank RNG、数据游标与原学习率日程，记录父 checkpoint SHA256 和父源码；
只允许更换输出目录、`model.self_attention_backend` 与 `abort_grad_norm`，其他配方、数据、运行环境和卡数必须一致。
这不等同于只加载权重的 `initialization=checkpoint`，也不放宽普通 `--resume` 的源码一致性检查。
`self_attention_backend=centered_fp32` 对短 self-attention 使用显式 FP32 softmax/反向和均值 key 中心化，
保持数学目标和参数结构，避免融合反向从大数值 LSE 重建概率；BF16 投影、cross-attention 和块编译保留。
该路径禁用 FP32 matmul 的 TF32，评估加载时同样生效。`abort_grad_norm>0` 在更新前停止异常的大梯度；
日志同时记录窗口内最大梯度范数，调度器遇到数值故障不自动从 `last.pt` 重试。

## 当前边界

- `configs/hand_spec.template.json` 是未标定模板。实际 Sharpa URDF、22 个关节顺序、21 个关键点 link/局部坐标、刚性掌部锚点、掌面射线区域和固定参考长度需要真实数值。
- 当前 Data v1 采用你最新指定的**完整 mesh 表面积均匀采样**；不是可见表面的单视角渲染协议。相机信息保留在场景 JSON，不参与本版采样。
- FlowMap 使用训练 split 条件下冻结 FM 的真实 ODE 轨迹，默认 Heun32，并混入完整 `(0,1)` 区间；学生有独立参数。
- 碰撞与抓稳结果由外部评估器提供。没有外部结果时物理成功率返回 `null`，不会把 IK 成功当作物理成功。

## 验证

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
```

训练循环软件测试禁用 `optimizer.step`。MinkowskiEngine 不可用时，仅真实稀疏网络测试跳过；其他模型测试使用限定在 `tests/` 的编码器替身。验证记录保存在外层 `../outputs/`；模型与优化细节见 [模型 v1](docs/model-v1.md)。

### Sharpa 1000 条数据与实验展示

基于现有 Dexonomy 物体与 Shadow 抓姿做 Sharpa 显式重定向，保持 22 DOF 与 21 个诊断关键点。数据来源、尺度、筛选与划分见 [data 说明](docs/data.md)。[Tailscale 数据浏览网站](http://100.91.234.79:8766) 展示真实 mesh、重定向骨架、检查结果与训练日志。运动学 / 几何检查通过不代表物理抓取成功。

## 配置与实验记录

`configs/` 保留 FM、FlowMap 通用配置、八卡 FM 配置、手模型模板、参考索引，以及 `ablations/` 中六个已定义结构的对照配置。通用配置需填写实际数据路径；各配置的初始化方式以 `initialization` 字段为准。新增具体运行配置与日志放入外层 `../outputs/` 的对应实验目录，配置路径按配置文件所在目录解析。

通过 `model.variant` 选择 `full`、`full_h_less`、`object_only`、`object_only_h_less`、`object_direction_adaln` 或 `object_direction_adaln_h_less`。省略时默认为 `full`，兼容原有模型权重。各版本的分支、方向注入和输入要求见 [模型说明](docs/model-v1.md#对照实验版本)。例如，`python -m dexgrasp plan --config configs/ablations/object_direction_adaln_h_less.json` 只检查配置和依赖，不启动训练。六份对照配置统一随机初始化、数据划分和训练预算，输出路径互相独立；正式八卡运行使用 `scripts/train_fm_ddp.py`。

单卡与多卡训练每 1,000 个 optimizer step 保留 `step_001000.pt`、`step_002000.pt` 等完整检查点（模型、优化器和恢复状态），训练结束时也保留最终版本；`last.pt` 继续指向最新保存结果。历史检查点不自动清理。保存使用原子替换与同目录硬链接，更新 `last.pt` 不会覆盖已归档版本。这里按 optimizer step 计数，不按遍历数据集的 epoch 计数。

对于已启动、仍使用旧保存逻辑的训练，可以运行 `python scripts/archive_live_checkpoints.py --checkpoint <last.pt> --pid <训练进程 PID>`：每两秒检查一次已完成的原子保存，按检查点内的步数归档，训练进程退出后做最后检查并结束。它不修改训练状态或源版本信息；旧进程的严格断点恢复仍需匹配其记录的源代码版本。

历史合成过拟合、32 条排错及批量／续训配置已移出代码仓库，清理时的临时归档和旧合成实验输出已删除。现有 Sharpa 实验的配置身份、指标与评估结果仍保留在对应运行目录。

`dexgrasp/train/` 实现训练，`dexgrasp/test/` 实现生成与评估，根目录 `tests/` 验证软件行为。`licenses/` 与 `THIRD_PARTY_NOTICES.md` 保存第三方代码许可证和来源。

### 正式 packed checkpoint 的运动学评估

`scripts/evaluate_formal.py` 支持正式 object-only、h-less FM 权重。先冻结按物体均衡的
train 诊断或 test 条件清单，再对各 checkpoint 使用相同条件、噪声和 Euler 步数：

```bash
.venv/bin/python scripts/evaluate_formal.py prepare --root <formal-packed-root> \
  --output <selection.json> --split test --conditions 4
.venv/bin/python scripts/evaluate_formal.py run --checkpoint <step_XXXXXX.pt> \
  --selection <selection.json> --output <new-output-directory> \
  --device cuda:0 --candidates 8 --batch-size 8 --workers 8 --steps 20
```

输出平均关键点 L2 误差、每姿态最大点误差的均值、IK Fit Success Rate（最大点误差 ≤5 mm
且关节合法）、官方手腕 +Zh 的方向夹角、条件内 22 关节角的样本标准差均值，以及生成与 IK 耗时。
IK 从关节限位中点初始化，不借用 GT q；求解器收敛状态与现有 RMSE≤3 mm 判据另行保存。
正式旧 hand_spec 缺少 direction_condition 时，仅在评估内存中明确官方 +Zh 的轴定义，
选择阶段逐条核对源数据 d=R[:,1]、归一化和样本身份，不改冻结数据。

同一物体内按条件等权、物体间等权聚合；失败保留在通过率分母中，连续误差有覆盖率，
不足两个可解码/IK 通过候选时多样性为 null。置信区间按资产组 bootstrap；不代表独立训练种子的波动。
条件选择不使用模型得分，test 不用于选择 checkpoint。生成计时使用预热后的 CUDA 同步，
包括编码器与采样，不包括加载、H2D、IK；默认 BF16 autocast、FP32 ODE 累加、关闭编译。
输出目录保存 protocol.json、逐候选 JSONL、原始关键点与噪声数组、summary.json。
`evaluate_formal.py compare --runs <run-a> <run-b> ... --output <new-summary-directory>`
校验条件清单、噪声、数据身份和指标协议一致且候选数完整后，导出 comparison.csv 与 comparison.json。
这是运动学评估，穿透、接触和物理成功率未测量，不以 IK 成功替代抓稳。

## 通用 DDP 与性能核验

`scripts/train_fm_ddp.py` 支持单机任意卡数；`batch_size` 是**每卡、每个 microbatch** 的样本数。
全局 batch = `batch_size × WORLD_SIZE × accumulation`。当前两个 object-only 对照固定每卡 1280、累积 1：双卡全局 batch 是 2560，四卡是 5120；
改变每卡 batch 也会改变优化器更新的全局 batch，不能只当成无关的性能选项。两种模型对照应使用相同的卡数、batch 和累积次数。

默认 `data_loading="stream"`，按批读取 HDF5，通过有界 worker 预取和 pinned memory 传输；
训练索引用紧凑整数数组保存，HDF5 按 shard 批量读取并恢复采样顺序。预取不推进检查点里的已训练游标。
`data_loading="gpu_cache"` 仅供能完整装入单卡的小数据集诊断；百万级数据使用 stream。
此入口使用当前 `data-v1` 数据协议，新的原始数据包需要先适配和冻结 train/test；不能把尚未适配的归档目录直接作为 dataset_root。

可选 `fps_backend="triton"` 使用融合 FPS，保留最远质心起点、最小索引 tie-break 和不重复选择规则。
`compile_transformer=true` 仅编译 Transformer block；动态稀疏骨干保留原实现，参数名和检查点结构不变。
编译启动开销不计入稳态吞吐。`sparse_convolution_mode` 可选 default/direct/copy/copy_hidden 用于实测调优；
不同点云的最优值可能不同，不能假定强制 GEMM 路径必然更快。核验环境中 default 最快。

FM 训练使用随机高斯噪声和均匀时间的直线路径，以 21×3 关键点速度 MSE 为目标；物体点云和关键点以物体质心居中并按 0.2 m 缩放（以配置为准）。
全参数训练保留 MinkUNet14 FP32、Transformer BF16、SyncBatchNorm、梯度裁剪和跨卡梯度平均。
累积时仅最后一个 microbatch 同步 DDP 梯度，但 SyncBatchNorm 每个 microbatch 都同步；因此累积不等价于一次大 batch 的 BN 统计。
仅 train 用于优化；test 不用于选 checkpoint。学习率固定，没有隐式调度器或自动按 batch 缩放学习率。

`optimizer="adamw"` 使用 fused AdamW（CUDA）；`optimizer="muon"` 使用 PyTorch v2.9.0 Muon 兼容实现加辅助 AdamW。
Muon 仅作用于隐藏 nn.Linear 权重；稀疏卷积、关键点输入投影、输出头、embedding、bias 和 norm 使用 AdamW。
默认 Muon 参数是 lr=0.02、momentum=0.95、5 次 Newton–Schulz、original LR 调整；辅助 AdamW 使用配置中的 lr。
两组参数和各自状态一起保存；改变分组、源码、配置、数据身份或 world size 会拒绝严格恢复。
严格恢复指状态与实验身份一致：模型和优化器完整恢复、每个 rank 的 RNG 恢复、按已消费 microbatch 恢复采样。
MinkowskiEngine CUDA 稀疏计算的独立重复运行并不逐位一致，因此不承诺恢复后的浮点训练轨迹与一次连续运行逐位一致。
每 1000 optimizer step 和最终 step 保留独立检查点，last.pt 是恢复入口。

性能脚本例子（需要当前项目环境及其可用 NCCL runtime）：

```bash
OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4 \
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_ddp.py --config <config.json> --output <result.json> --updates
```

`benchmark_ddp.py` 测量真实骨干、DDP 同步、流式输入和可选的真实优化器更新；每轮取最慢 rank 的耗时，检查所有参数梯度有限且跨 rank 完全一致。
默认不更新权重，`--updates` 仅作短程软件核验，不保存正式训练模型。GPU 算子回归测试：

```bash
DEXGRASP_GPU_TESTS=1 PYTHONPATH=tests .venv/bin/python -m unittest test_execution -v
```

显存选项 `checkpoint_direction_bias=true` 仅重算方向几何偏置 MLP 的激活，不重算整个 Transformer；
该 MLP 没有随机操作，保留输出及梯度语义。AdaLN 版本没有该分支，设置同样选项即可。
`select_tokens_before_slice=true` 将 FPS 索引与稀疏逆映射组合，直接提取 128 个点的特征，避免先恢复全部 1024 点的 512 维特征。
这两个选项与上述编译、融合 FPS 可独立核验；不要把单算子的加速倍数当成整网加速倍数。

百万级数据应先在进程组外做一次完整审计，再设置配置的 `data_audit` 指向审计结果：

```bash
.venv/bin/python scripts/audit_training_data.py --dataset-root <data-v1-root> --output <audit.json>
```

DDP 会重新检查审计代码、手模型、划分和所有 shard 的 SHA-256，不会仅凭 READY 文件跳过内容验证。
未经完整审计、协议适配或划分冻结的新归档不能直接训练。

本机 5090 的双卡验证使用以下进程级通信设置（没有改系统或共享环境）：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4 \
NCCL_P2P_DISABLE=1 NCCL_CUMEM_HOST_ENABLE=0 \
LD_PRELOAD="$PWD/.venv/nccl-runtime/nvidia/nccl/lib/libnccl.so.2" \
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/train_fm_ddp.py --config <validated-config.json>
```

四卡运行把 nproc_per_node 改成 4；两个四卡任务分别用 CUDA_VISIBLE_DEVICES=0,1,2,3 与 4,5,6,7。
环境和数据准备完成前只运行 plan 或有界 benchmark；上面的训练命令会实际更新并保存权重。

最终双卡开发核验将两个 object-only 版本都固定为每卡 1280、累积 1；优先 fused AdamW。
相同 50 步稳态测试中，进程级 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
将几何偏置版的 PyTorch 峰值预留显存从 29.91 降到 25.25 GiB，方向 AdaLN 版从 28.57 降到 24.01 GiB，
吞吐分别约 4194 / 4580 samples/s，较默认分配器变化不足 0.5%。实际分配峰值分别约 25.03 / 23.92 GiB。
这些数值包含真实反向与 AdamW 更新，来自旧 800 条 train 样本的重复采样，不包含编译启动和 checkpoint 写盘；
不能作为新百万级数据或四卡吞吐的实测。新数据适配后需再次验证相同 batch 的显存和 I/O。
使用此选项时将它加在上面的启动命令前；恢复必须保持同一 allocator 配置。
[PyTorch 2.7 分配器说明](https://docs.pytorch.org/docs/2.7/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)。

## 新数据 I/O profiling 与可复用训练表

`dataset_format` 默认 `data_v1`；`object_corpus` 是原始 object-only 数据的参考读取器适配层，
`packed` 是独立于具体数据包、样本数和模型权重的只读训练表。后两者目前供有界 benchmark/profile 使用：
新语料的研究划分尚未冻结，正式 DDP 入口会明确拒绝使用诊断划分，不能把性能测量当作正式训练。

训练表通过 `.npy` memory mapping 按批读取，使用观测索引共享点云，支持重复/乱序样本并保持原采样顺序。
它只预计算确定性的归一化，不存储网络特征、不改变随机采样、不把整套数据复制到每卡显存，也不要求整套数据常驻 RAM。
后续数据适配器可以发布相同 `packed.json` 数组/索引契约；训练表读取器不包含包名或数据量特例。
源文件 SHA-256、变换源码、数组形状/类型/哈希和划分身份随产物保存；源数据或确定性预处理变化后需构建新目录。
未来在线随机增强应在读取之后执行，不能烘焙成这一层静态缓存。

当前原始数据适配器可这样构建训练表（输出目录必须不存在）：

```bash
.venv/bin/python scripts/pack_object_corpus.py --source <raw-corpus-root> --output <new-packed-root>
```

它选择一个 frozen base 及有排重证明的增量，排除重叠首批，完整校验源清单，保留真实 h32 和 +Z 方向。
生成的 object-hash train/test 仅用于性能诊断；不会自行宣布研究协议冻结。
`raw-index.json` 与 `packed.json` 共享相同样本索引，可用相同配置对比原始 reader 与训练表。
原始 reader 仍读取提供方的 HDF5/NPZ，因此它衡量的是现有参考管线；训练表的收益包括减少文件打开、解压/解码和逐样本拼装。

`benchmark_input.py` 支持 torchrun 双进程，隔离测量 DataLoader 首批延迟、后续吞吐、进程 RSS 与读取/缺页计数。
`profile_ddp.py` 单独输出稳态阶段计时和 CUDA trace，覆盖 input wait、H2D、forward、backward/DDP、clip/optimizer，
并在结束检查各 rank 完整模型与 BN 状态一致、所有参数有梯度。Trace 和其额外开销不计入稳态吞吐。
CPU/GPU profiling 工具只做有限步真实更新，不保存正式模型；不要同时运行两组性能测试。
输入测速不清理系统 page cache，首次进程读取不能直接称为“冷盘吞吐”。

新全量＋增量（4,328,729 条）已在独立双卡 dev 上完成实际更新 profiling。
同为每卡 batch 1280、2 workers、预热 5 步、对齐前 30 个测量步，原始 reader 与训练表分别为：

| 版本 | 原始 reader samples/s | packed samples/s | 倍率 |
| --- | ---: | ---: | ---: |
| object_only_h_less | 1076 | 4691 | 4.36× |
| object_direction_adaln_h_less | 1063 | 5229 | 4.92× |

训练表每卡峰值实际分配显存（包含预热，55 次更新）分别约 22.59 / 21.48 GiB。
原始 reader 调至每卡 8 workers 后，在预热缓存条件下也能基本喂满 GPU；因此上述倍率是同 2 workers 的管线改进，
不能解释为对已充分并行的热缓存 reader 仍有 4–5 倍整网加速。
只回收本次任务的本地训练文件页缓存后，packed 2 workers 双进程吞吐约 7853 samples/s（含启动），
原始 reader 8 workers 约 1636 samples/s。读取字节计数确认发生实际磁盘读取，没有清理系统全局 page cache。
本语料的训练表约 1.56 GiB，能放入机器 page cache；尚未实测超过 RAM 容量的数据规模，不能保证任意存储上的同等倍率。

一次性源校验与训练表构建约 58 秒；首次传输不是每轮训练成本。checkpoint 测量使用约 779 MB 完整模型＋优化器状态：
GPU→CPU 约 0.41 秒，共享盘序列化写入及 fsync 约 2.40 秒，按每 1000 步保存摊销约 0.5%，保留原有原子保存与永久归档。
完整阶段计时、两卡 trace、冷/热读取、worker 对照、校验与配置保存在外层 `outputs/dev_newdata_profile_20261111/`。

## 崩溃前已保存 checkpoint 的测评

`scripts/evaluate_checkpoint_sweep.py prepare` 盘点正式 A/B 的所有 `step_*.pt`，记录权重 SHA-256，
核对 `last.pt` 是否为已盘点版本，冻结选择清单与评估源码。相同权重、选择、采样和指标代码的完整结果可以复用。
它只读取权重，不训练、不清理历史版本，也不按 test 指标选择 checkpoint。

```bash
.venv/bin/python scripts/evaluate_checkpoint_sweep.py prepare \
  --checkpoints ../checkpoints/sharpa_formal_20261111 \
  --selection ../outputs/ckpt_metrics_20261112/selection.json \
  --reuse-root ../outputs/ckpt_metrics_20261112 --output <new-sweep-directory>
.venv/bin/python scripts/evaluate_checkpoint_sweep.py run \
  --plan-path <new-sweep-directory>/plan.json --gpus 0 1 2 3 --workers 12
```

运行前核对 GPU 是否可用；`workers` 是每个并行任务的 IK worker 数。调度器使用文件锁防止重复启动，
原子更新 `status.json`，为每个任务保留日志，全部成功后生成 `comparison/comparison.csv` 和 JSON。
中断后可用同一命令恢复：完整结果保留复用，不完整结果改名留档；已有活跃 worker 时拒绝重复运行。
使用持久进程启动方式可让任务在终端断开后继续。

历史归档 `../outputs/ckpt_sweep_20261112/plan.json` 固定了 41 个独立版本（A 18、B 23），
407 个 test 物体，每物体 4 个条件、每条件 8 个候选，共 533,984 个候选。
当前按 `../checkpoints/sharpa_formal_20261111/evaluation_policy.json` 排除训练崩溃后的权重：
A 仅保留 ≤40k、B ≤90k，共 26 个、338,624 个候选。排除依据是训练异常记录，不按 test 指标筛选。
当前 `evaluate_checkpoint_sweep.py` 的 prepare/run、formal run/compare 与 physics run 遵循该策略；
后续执行使用当前脚本入口，历史源码快照仅保留审计，不改写旧计划、权重或原始结果。
网站的只读展示适配器位于 `../site/`；生成耗时为当前设备上的批量摊销测量，不是单请求延迟。

## 复用 inference 的几何与物理评估

`scripts/evaluate_physics.py` 接在 `evaluate_formal.py` 之后，读取保存的 `candidates.jsonl`
中的 `R/t/q`，恢复逐条件物体中心与缩放，不再 inference。准备碰撞资产、评估、汇总可分别执行：

```bash
.venv/bin/python scripts/evaluate_physics.py prepare \
  --selection ../outputs/ckpt_sweep_20261112/selection.json \
  --assets ../data/eval_physics_sharpa_v1 --workers 12
.venv/bin/python scripts/evaluate_physics.py run \
  --selection ../outputs/ckpt_sweep_20261112/selection.json \
  --assets ../data/eval_physics_sharpa_v1 \
  --runs ../outputs/ckpt_metrics_20261112/A_040000 ../outputs/ckpt_metrics_20261112/B_090000 \
  --output ../outputs/physics_eval_20261113 --workers 24
```

可用 `--limit 4` 在独立目录冒烟测试，或 `--geometry-only` 仅计算几何指标。正式结果要求
`full_selection=true`；限制条件数的结果不能当作完整 test。中断后使用同一命令恢复已完成的条件；
输入、协议、源码或资产变化时拒绝混用。完整结果是 `summary.json`、`comparison.csv`、
`candidates.jsonl`，逐条件明细在 `chunks/`，完成状态在 `status.json`。

全量 41 个 checkpoint 的补测位于 `../outputs/physics_all_20261113`，输入清单在
`../outputs/physics_all_20261113_plan.json`，复用原 533,984 个生成候选，另测 1,628 个 GT 姿态。
全量批次统一使用 128 MiB MuJoCo arena；静态几何只执行运动学与碰撞查询，先筛除严重穿透，
再进入动力学，避免崩溃后的重叠手姿耗尽约束求解内存。源码、运行参数均保存在该批次协议中。
网站 `/#test` 展示 26 个崩溃前 checkpoint 的全部指标、GT 基线、逐候选受力诊断及 CSV；正式可比 GSR/OSR 仍未建立。

指标参考 [KPGrasp §5.1](https://arxiv.org/html/2606.09314v1#S5.SS1)、
[DexGraspBench 的几何定义](https://github.com/JYChen18/DexGraspBench/blob/d9ea6cf282de1f463c20fa54b4f68d7025bad40e/src/task/eval_func/base.py)
和 [PCA 定义](https://github.com/JYChen18/DexGraspBench/blob/d9ea6cf282de1f463c20fa54b4f68d7025bad40e/src/task/statistic.py)。
实现为独立代码，物理协议名称是 `sharpa-single-pose-force-test-v1`。

| 指标 | 定义与边界 |
| --- | --- |
| 手物 PD / 自穿透，mm | MuJoCo 碰撞接触中最深负距离，非关键点/点云穿透；无负距离为 0 |
| 接触距离 / CDC，mm | 每根手指最近距离的正部均值 / 五指距离极差；搜索截断 10 mm，CDC 保留负穿透 |
| 接触链接数 | 与物体距离绝对值小于 2 mm 的不同手指链接数，不含手掌 |
| 单姿态 GSR / OSR | 全部候选的六方向受力通过率 / 至少一个候选通过的物体比例 |
| PCA 多样性，% | 未标准化的物体坐标平移(m)、官方手腕旋转向量(rad)、22 关节(rad)；累计前 1–5 PC 方差占比，越低越分散 |
| IK / 方向 / 耗时 | 合并原评估结果；IK 成功与物理成功分别统计，并另报二者同时成功率 |

碰撞模型使用官方 Sharpa XML 的碰撞体、位置控制器、惯性和自碰撞排除项；每个条件都验证其 FK 与
训练 URDF 一致（最大关键点偏差必须低于 0.1 mm）。物体使用归档完整网格的 CoACD 分解，按该条件真实
scale 缩放；这是碰撞几何近似，不用整个凹物体的单一凸包替代。下载资产与许可、分解参数和哈希放在 data。
评估协议记录代码版本/源码快照、输入与权重身份、冻结 test 划分、资产哈希、MuJoCo/CoACD 版本。

物理测试固定手腕，原生位置控制器保持预测关节角，物体质量 0.1 kg，关闭重力和桌面，摩擦 0.6/0.02，
每方向独立复位，先稳定 1 s，再施加 1 N 外力 2 s，每 0.2 s 检查位移 < 5 cm、转角 < 15°。
初始手物或自穿透 > 10 mm 计失败，六方向全部通过才成功，模拟告警也计失败。
当前模型只有单姿态，**没有生成 pregrasp/squeeze**；这与 KPGrasp 的 Shadow 三阶段、候选排序和数据划分不同，
因此单姿态 GSR/OSR 是 Sharpa 诊断，不能直接与论文成功率比较，也不测桌面接近/抬升。
同条件 GT 单姿态一起运行，用于揭示协议差异；其源数据的成功标签不自动转移为本测试成功。

聚合先对条件、再对物体等权，95% CI 对 asset group 做 2000 次 bootstrap。
失败不从成功率分母删除，连续指标附覆盖率；PCA 同时报全部可解码姿态与受力成功子集，样本不足或零方差为 null。
CDC 为 0 也可能是五指全部远离物体（均被截断），必须与接触距离、接触数一起读。
本轮 A40k/B90k 来自训练失稳前的保存节点，依据与训练日志哈希记录在外层
`outputs/physics_eval_20261113_plan.json`，不按 test 质量选择最优 checkpoint。

2026-11-13 已完成这两个 checkpoint 各 13,024 个候选及 1,628 个 GT 单姿态的完整评估，
结果在外层 `outputs/physics_eval_20261113/`，独立核验记录为 `verification.json`。
A40k/B90k 的手物 PD 为 2.712/2.085 mm，自穿透为 0.0526/0.0224 mm，CDC 为 8.214/7.466 mm。
单姿态受力通过率为 1.643%/1.674%，同条件 GT 只有 0.676%；这说明该执行协议尚不能承接源数据的
物理成功标签，不能将这些低通过率归因于生成模型。正式可比的 GSR/OSR 仍需补齐并验证抓握闭合和
pregrasp/squeeze 执行协议。GT 每条件 1 个、模型每条件 8 个，其 OSR 候选预算也不同。
18 项相关软件测试、实际远离物体负对照、力/质量数值校验与续跑测试通过；FK 最大偏差小于 0.0001 mm。

## 2026-11-11 正式四组对照

用户授权正式训练：每组 4×RTX 5090，每卡 1280，累积 1，全局 5120，200,000 optimizer steps。
A/B 为几何方向偏置/AdaLN + AdamW；C/D 为相同两结构 + Muon/辅助 AdamW。
按最新修正，GPU 0–3 顺序 A→D，GPU 4–7 顺序 B→C；后继随机初始化，不继承前驱权重。
四组都使用基准 lr=muon_lr=2e-4，前 5k 从 4e-5 线性预热，随后 cosine 至 2e-5；
weight decay=0、grad_clip=1、seed=42。每 5k 永久保留 checkpoint，结束保留 step_200000。
`last.pt` 原子替换，学习率由已完成的 optimizer step 确定，混合优化器两分支共同恢复。

配置源位于 `configs/formal_20261111/`。已验证代码和启动配置另行冻结在外层
`outputs/formal_schedule_20261111/{source,configs}/`；训练从冻结源码执行，后续工作区修改不会影响它。
`plan.json` 定义两条队列；`lane_AD.json`、`lane_BC.json` 与 `supervisor.json` 记录实际状态。
后台调度器是持久进程，不是系统启动服务；训练异常仅在已有可恢复 checkpoint 时最多重试三次，
失败时保留日志并阻止该队列后继，其它队列继续。不能把已提交调度当成四组已完成。

在外层工作区，用现有项目 Python 运行冻结的 `source/scripts/schedule_experiments.py --plan <plan.json>`。
启动前核对 supervisor 和训练进程，勿启动第二个管理器。持久锁防止重复调度器；已有活跃 worker 时拒绝重复启动。
恢复需保持冻结代码、配置、数据、软件环境和每组四卡不变。原始归档仍在 data/archives；本机 /var/tmp 训练表不是持久备份。
冻结划分和打包元数据备份于外层 `data/sharpa_formal_20261111/`；`local_cache/` 指向实际训练表。
面板复用工作区 site，读取 `outputs/sharpa_formal_20261111`，展示四组排队状态、实际 LR、loss、吞吐和 GPU 分配。
