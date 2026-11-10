# 1000 条 Sharpa 重定向诊断

本实验保持 Sharpa 手型，利用 Dexonomy 现有 Shadow 抓姿与对应 DGN 物体 mesh，通过显式运动学重定向生成监督标签。数据及生成的关键点适合检查训练链路；未把上游 Shadow 仿真成功标签继承为 Sharpa 成功标签。

## 数据来源与尺度

固定 Hugging Face 数据版本 `JiayiChenPKU/Dexonomy@aa432b9f9d7a7355e9ab4fa881154a5f1ac4f881`。DGN 物体归档与 Shadow 抓姿归档均与上游 SHA-256 校验。数据遵守 CC BY-NC 4.0。

- DGN 归档 SHA-256：`dda297aee2d2779142da98d4037edd89616286129c85eb7395b96c46e03a6231`。
- Shadow 归档 SHA-256：`9aebdbc564cc6f65399befbb5984aa58114a8d09292e34a83493b1976d416b14`。
- 基础尺度来自原始 scene config，逐条乘以 `scene_scale[i]`，不是把长度为 N 的数组当作 xyz 各向异性缩放。原场景物体姿态也参与 mesh 变换。
- 每条样本保存源 archive/member、原始行号、scene config、物体 mesh、实际尺度、拟合误差和检查结果。使用完整 mesh 均匀采样，不冒充真实相机点云。

初次仅使用归档前缀中的 7,767 条抓姿，80 个物体共 380 条通过；记录在 `outputs/sharpa_dexonomy_1k/prefix_screen/`。通过率不足后，补齐相同 80 个物体的全部可用浮空抓型，保持筛选门槛不变。物体候选池仍受归档顺序影响，并非代表性随机数据。

## 重定向和筛选

Shadow root 为 xyz + wxyz 四元数，之后为原 XML 遍历顺序的 22 个手指关节。用 MuJoCo 做源 FK。显式定义腕部和每指四点，共 21 个对应解剖标记；这些点不是 KPGrasp 官方球面标记。

保持源掌面朝向，Sharpa 的局部 x/y/z 对应 Shadow 的 x/z/-y。优化 22 个有界关节和不超过 ±30 mm 的根平移，拟合对应关键点，指尖残差权重为 2；不会直接复制不同手型的关节角。Sharpa 仍使用官方 URDF 和上一轮项目定义的诊断标记及掌面平面；这不是硬件表面标定。

筛选要求：优化收敛；关节限位；平均关键点误差 ≤15 mm；最大指尖误差 ≤25 mm；源 mesh 闭合且绕序一致；64 条掌面射线全部命中且起点不在物体内部；21 个骨架关键点的最大物体内部深度 ≤5 mm。骨架检查不覆盖完整手部 mesh、指节中段或自碰撞。未做物理抓取仿真。

从确定性打乱后的候选物体中，已选出前 20 个各有 50 条通过的物体，形成 800/100/100 的 train/val/test，物体级无交叉。去重依据原始 root/joints 与尺度的精确 float32 内容。原始任务为 floating；为了满足模型的环境通道，增加 z=-0.4 m 的远离手与物体的人工地面，仅作为输入上下文，不赋予桌面抓取成功语义。

## 初始化、冻结与存储

FM 随机初始化，全参数训练，包括物体 / 场景编码器、Transformer、时间与条件嵌入、输出头。旧的 12 条合成诊断 checkpoint 不作为本轮初始化。FlowMap 在本轮 FM 完成后复制教师权重，新增结束时间输入置零；教师参数与 BN 统计冻结，学生全参数训练，采用 Euler 20 步教师。

- 数据：`data/sharpa_dexonomy_1k_v1/`。
- checkpoint：`checkpoints/sharpa_dexonomy_1k_v1/<run>/{best,last}.pt`。
- 配置：`configs/sharpa_1k_fm.json`，后续运行配置按 run 单独存储。
- 日志与协议：`outputs/sharpa_dexonomy_1k/`。

Checkpoint 包括模型、优化器、RNG、配置、代码版本及文件哈希、运行时、数据划分与手模型哈希。恢复必须匹配这些身份；改配置时开启新 run。

过拟合选择依据是全部训练样本的固定种子生成终点 MPJPE，目标 <5 mm。独立种子评估遍历全部样本，不筛选最佳候选，报告 train/val/test 的均值与 P95。测试集没有参与选择 checkpoint。该指标与物理抓取成功严格分开；未达目标也必须保留实测结果。

## 网站

`http://100.91.234.79:8766`，连接本 tailnet 后访问。展示真实 mesh、原始 Shadow 与 Sharpa 重定向骨架、样本误差与射线检查、训练曲线及测得的评估结果。只读服务绑定 localhost，利用当前 Tailscale userspace 入站转发，无公网 Funnel。

网站代码位于 `site/`，启动方式见其 README。所有图形都来自数据或日志。设计概念图仅作为视觉参考，不作为实验结果。浏览器 QA、图像对比、实验结果与完整来源清单在外层 outputs/data 目录。

## 数据验收结果

最终恰好 1000 条，20 个物体，800 / 100 / 100。源候选池 81,076 条；每个候选物体获得 50 条通过样本即停止，所以并未对全池逐条完成 IK。共收集 3,899 条合格候选，从中按预定物体顺序取 1000 条。重定向 MPJPE 中位数 6.398 mm、P95 7.499 mm。所有 1000 条均通过独立 MuJoCo FK 复核，最大误差 9.463e-9 m。20 个源 OBJ 的精确字节哈希不同；此检查不涵盖近似重复几何。

另设固定 32 条排错对照：每个训练物体取存储顺序前两条，保留原来的 100 条 val / 100 条 test，独立随机初始化。配置 `configs/sharpa_control32_fm.json`，以 `CUDA_VISIBLE_DEVICES=2` 映射为逻辑 `cuda:0`；该环境变量也是恢复运行的必要条件。MinkowskiEngine 在直接使用非默认逻辑 CUDA 设备时发现 sparse-matmul 设备不一致，因此用标准设备隔离方式执行；初次失败发生在 0 个 optimizer step，记录在 outputs 的 setup_failures 中。32 条对照结果不能冒充 800 条训练结果。

## 本轮最终结果

数据为 20 个物体 × 50 条重定向抓姿，800/100/100 按物体划分。以下均为 8 个独立噪声种子的均值，无候选排序。

| 实验 | 训练样本 | train / mm | val / mm | test / mm | 5 mm 目标 |
|---|---:|---:|---:|---:|---|
| fm | 800 | 29.675 | 72.797 | 85.346 | 未达到 |
| flowmap | 800 | 30.548 | 72.630 | 85.546 | 未达到 |
| control32_refine | 32 | 3.659 | 80.727 | 109.143 | 达到 |

32 条对照只证明小规模训练链路具备记忆能力，不能代表 800 条训练集已充分拟合，也不能证明未见物体的泛化能力。FM 教师尚未充分拟合，当前 FlowMap 蒸馏同样未达到主实验目标。

FM 从随机权重开始，91,662,659 个参数全部训练。FlowMap 从本轮 FM best 初始化，91,728,195 个学生参数全部训练；教师 91,662,659 个参数及 BN 统计冻结。32 条对照先训练 4000 步，再从其 best 初始化，以 3e-5 学习率和新优化器继续 2400 步。

Checkpoint 在 checkpoints/sharpa_dexonomy_1k_v1/<run>/{best,last}.pt，含优化器、RNG、配置和身份哈希。summary.json 记录已重新核验的 checkpoint SHA-256；各 run 的 provenance.json 及训练前代码归档记录实际版本。

1000 条标签均通过独立 MuJoCo FK 验证；41 项软件测试通过。网页已通过桌面/手机交互测试。没有完整手部碰撞、自碰撞、物理抓取或硬件成功率评估。

网站：http://100.91.234.79:8766（需连接 tailnet；本机 Tailscale 网络栈验证，未从第二台设备复核）。服务持续运行，但尚未设置开机启动。

## 八卡 FM 训练（global batch 2048）

`configs/sharpa_1k_fm_ddp8_batch2048.json` 的 `batch_size=256` 表示每卡批量，8 个 DDP rank 每步共处理 2048 条，梯度累积为 1。只有 800 条训练数据，因此用确定性的连续 shuffled epoch 流分配每个 rank 的采样位置，跨 epoch 重复样本并使用独立噪声和时间；不引入 val/test。每步均为完整 2048 条，没有末批缩小。

从 `fm_batch256_continue19k/last.pt` 的第 1000 步模型权重继续，AdamW 重新初始化，学习率保持 3e-5。模型及编码器全部训练；Transformer 使用 BF16，稀疏编码器保留 FP32，允许 TF32。使用 SyncBatchNorm 同步稀疏特征统计。

为减少延迟，所有固定训练输入和监督标签预装到 GPU，FPS 索引预计算一次；缓存前后前向结果与梯度等价的软件测试已通过。启用 fused AdamW、DDP 梯度 bucket view，loss 每 10 步归约和记录一次。训练过程中不做重建评估。Checkpoint 保存未包装的模型键名、优化器、全部 rank 的 RNG、全局采样步数、配置和来源哈希；支持相同 world size 的恢复，并兼容现有单卡推理读取。

启动（内层仓库目录）：

```bash
.venv/bin/python -m pip install --no-deps --target .venv/nccl-runtime nvidia-nccl-cu12==2.27.7
LD_PRELOAD="$PWD/.venv/nccl-runtime/nvidia/nccl/lib/libnccl.so.2" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NCCL_CUMEM_HOST_ENABLE=0 NCCL_P2P_DISABLE=1 .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 scripts/train_fm_ddp.py --config configs/sharpa_1k_fm_ddp8_batch2048.json
```

恢复时增加 `--resume ../checkpoints/sharpa_dexonomy_1k_v1/fm_ddp8_batch2048/last.pt`。该机器实测 GPU P2P 不可用，默认 NCCL cuMem host 分配路径报错；上述两个进程级环境变量改用传统共享主机内存。系统 NCCL 2.25.1 随后仍出现 `enqueue.cc:1500 invalid argument`，与 [NVIDIA 记录的 SM120 兼容问题](https://github.com/NVIDIA/TensorRT-LLM/issues/4238) 一致，因此使用项目内 NCCL 2.27.7 并通过 `LD_PRELOAD` 仅对训练进程生效，没有修改系统配置或共享环境。Checkpoint 记录实际加载的 NCCL 版本、动态库 SHA-256 和通信环境。每个 rank 在导入 Torch/ME 之前把对应物理卡隔离为逻辑 `cuda:0`，避免现有 ME 的非默认设备路径问题。

实现依据：[PyTorch 2.7 DDP](https://docs.pytorch.org/docs/2.7/generated/torch.nn.parallel.DistributedDataParallel.html)、[NCCL 环境变量](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-cumem-host-enable)。运行协议、初始启动失败日志和实测吞吐均保留在外层 `outputs/sharpa_dexonomy_1k/`。

八卡首段 20 步已完成：所有 rank 的模型及 BN buffer SHA-256 一致，保存后恢复训练成功。恢复后第 40–120 步实测平均 0.642 秒/步，约 3189 样本/秒。此时 nvidia-smi 每卡约 26.1–27.2 GiB；瞬时 GPU 利用率随同步阶段波动。软件测试合计 45 项通过。详细记录见 `fm_ddp8_batch2048_verification.json`、`fm_ddp8_batch2048/rank_consistency.json` 和 `fm_ddp8_batch2048/metrics.jsonl`。这些验证仅证明并行训练与恢复正常，生成误差等待训练完成后评估。
