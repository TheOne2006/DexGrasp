# 数据、训练与测试接口

所有命令从自有 `DexGrasp/` 仓库运行。`python` 指选定环境的解释器，不依赖 shell 自动激活。JSON 输出适合后续 agent 解析；退出码 0 表示命令成功，2 表示配置/数据阻塞或错误。`planned` 不表示训练完成，`reference_plan_only` 不表示参考环境已就绪。

## 数据协议

首批自有模型共享 `shadow28_xyz_m_3phase`：Shadow Hand，三个阶段依次 pregrasp/grasp/squeeze，每阶段 28×3 关键点，单位米。关键点名称和顺序取自本地 KPGrasp 的 `kinematics/ik.py`。这是第一版适配器的协议，不是对项目最终表示的决定；RTJ、Leap Hand 和 RL trajectory 必须增加各自的表示、转换和评测适配器，不能冒用此标识。

规范 NPZ（无 pickle）包含：

| 字段 | 形状 | 约定 |
| --- | --- | --- |
| point_cloud | N×3 | 与抓取姿态同一坐标系、已应用物体缩放，单位米 |
| keypoints | G×3×28×3 | G>0，同一物体场景的多个抓姿；纯测试点云可不含此字段 |

采样接口不将测试关键点输入模型；测试只有点云也可工作。训练时用完整点云均值居中，点云和关键点同时除以固定 `scale_m=0.1`；输出做逆变换。此归一化不使用验证/测试集统计量。训练按物体、该物体记录、记录内抓姿三级均匀抽样，采样分布与官方实现不完全相同。

输入 catalog 示例：

```json
[
  {"id":"DGN/a/scale008","object_id":"DGN/a","official_split":"train","storage":"npz","path":"../processed/a008.npz","scene_path":"../raw/object/DGN/scene_cfg/a/floating/scale008.npy"},
  {"id":"DGN/b/scale008","object_id":"DGN/b","official_split":"train","storage":"npz","path":"../processed/b008.npz"},
  {"id":"DGN/c/scale008","object_id":"DGN/c","official_split":"test","storage":"npz","path":"../processed/c008.npz"}
]
```

同一物体的不同尺寸、视角、抓型和姿态必须使用同一 `object_id`，跨数据集的相同几何资产也应先统一身份。工具能检测声明的 ID 泄漏、跨 split 复用 NPZ 和重叠 HDF5 行，不能仅凭文件名发现不同 ID 下复制的同一网格。正式下载后还需做网格身份去重。

```bash
python -m dexgrasp_lab prepare-catalog --catalog ../data/manifests/catalog.json --output ../data/manifests/dexonomy_kp.json --seed 0 --val-fraction 0.1
python -m dexgrasp_lab audit-data --manifest ../data/manifests/dexonomy_kp.json --deep
```

工具保留官方 test，从官方 train 物体中稳定划出验证集。同一 seed、不同输入排序产生相同物体划分。不会生成空的“已准备”manifest。深检查检查所有记录的形状/边界，并每记录读取一个确定性样本检查有限值；不是对所有抓姿的逐元素扫描。每次训练实际访问样本时也检查有限值。

## 两种准备入口

已有 KPGrasp 关键点 HDF5 时：

```bash
python -m dexgrasp_lab prepare-kpgrasp --root ../data/raw/dexonomy --output ../data/manifests/dexonomy_kp.json
```

输入结构严格遵照 KPGrasp `docs/DATA.md`，包括两个 collection、官方划分、`dex_kp_h5/index_obj_grasp.json`、shards 和 `vision_data/compl_pc.h5`。索引 `[offset,count,shard]` 按行惰性读取，不把整个 shard 放入 RAM。默认拒绝缺少索引或点云的物体，避免静默缩小测试集。

这个入口主要用于监督学习对照；每个物体记录可含多个 scale，当前不会猜测所抽样 scale 对应的物理 `scene_path`。需要完整物理评测时，用按 scene/scale 组织的 NPZ catalog，或另做经核对的场景适配器。未提供正确 scene_path 的采样会在 IK 导出前明确报错。

只有关节姿态时，先把三个 `*_qpos` 数组整理为无 pickle NPZ，每个 `(G,29)`，含 xyz+wxyz+22 joints。原始 Dexonomy 的字典 NPY 和作者机器上的绝对 `scene_path` 需要在数据下载后核实再转换；本轮没有猜测其路径映射。每份 NPZ 对应一个固定物体 scale。点云输入 NPY 是未缩放物体坐标，姿态平移已是米。

```bash
python -m dexgrasp_lab convert-qpos --source /absolute/poses.npz --cloud /absolute/pc.npy --output /absolute/scene008.npz --kpgrasp-root ../reference/KPGrasp --joint-order /absolute/input_joint_names.json --scale 0.08
```

`input_joint_names.json` 是输入姿态最后 22 维的真实关节名顺序；工具验证名字集合并重排到参考 XML 的顺序。不能直接假定不同数据源顺序一致。FK 使用 MuJoCo，输出仅为关键点，不证明物理成功。

## 训练接口

配置分离 `architecture`、`objective` 和 `manifest`。四份 JSON 构成 2×2 实验矩阵；更换数据用同一协议的 manifest，更换模型扩展 `models.py`，新生成目标扩展 `methods.py`，同时增加测试和配置校验。没有用任意模块导入或 shell 字符串做隐式分派。

默认单卡 FP32、microbatch=8、梯度累积=4，有效 batch=32；width=128、depth=2、1024 点。默认 10,000 updates 只是未执行的起步预算，不等价于论文训练量。无自动混合精度或 DDP；这些应在需要时单独验证。换 GPU 用环境变量 `CUDA_VISIBLE_DEVICES`，每个作业必须有不同输出目录。

Flow Matching：t=0 为高斯噪声，t=1 为数据，线性路径，回归 `x1-x0`，Euler 采样。Diffusion：1000 步 cosine 噪声日程，epsilon 回归，DDIM eta=0 采样。它不是对 DexLearn 的逐行复现。FM 的一大步 Euler 不等于 FlowMap/MeanFlow。

训练实现包含有限值检查、梯度裁剪、AdamW、梯度累积、固定验证噪声、独立测试划分、原子 checkpoint、best/last、RNG/optimizer 恢复。验证损失只用于同一目标内部选 checkpoint，不能直接比较 FM loss 和 diffusion loss 的数值。

记录 resolved config、manifest 内容/hash、代码 hash/Git 状态、PyTorch/GPU 信息、数据文件大小/mtime、每步 loss/耗时和显存。数据大小/mtime 不是内容完整性证明；正式数据版本和下载校验仍需补齐。恢复拒绝更改配置、manifest 或源码的运行；有意改配置属于新实验。

`train` 只有加 `--execute` 才执行。本轮不提供自动训练队列，也没有运行优化器更新。验收后应先运行小数据过拟合和实际断点恢复，再提交长作业。

## 模型 test：采样与评估

```bash
python -m dexgrasp_lab sample --checkpoint /absolute/best.pt --manifest /absolute/manifest.json --output /absolute/new_samples --steps 5 --candidates 10 --device cuda:0
```

默认打印计划；加 `--execute` 才采样。测试 manifest 必须与训练记录一致。输出包含 NPZ、样本元数据、checkpoint/manifest hash、seed 对应配置、NFE 和生成时间。没有 likelihood 排序；全部候选保留。时间包括点云编码，不含 IK 和物理仿真，也未做稳定性能基准的 warmup。

```bash
python -m dexgrasp_lab test --samples /absolute/new_samples
python -m dexgrasp_lab ik --samples /absolute/new_samples --output /absolute/new_ik --kpgrasp-root ../reference/KPGrasp
python -m dexgrasp_lab benchmark-plan --root ../reference/DexGraspBench --python /absolute/eval_env/bin/python --samples /absolute/new_ik --output /absolute/test_results --name trial
```

IK 仍需 `--execute`，依赖上游 solver 的 MuJoCo/mink/quadprog 等。每份样本必须有有效 scene_path；导出 qpos 为 xyz+wxyz+22 joints。`test`（兼容别名 `metrics`）只给关键点坐标标准差等描述值，物理成功率保持 null。IK 阈值成功不能代替抓取成功。

`benchmark-plan` 仅输出格式转换、MuJoCo 评测、统计三条 argv，不运行上游带清理操作的 shell 脚本。记录 main 分支协议：0.1kg、摩擦 [0.6,0.02]、最大初始穿透 0.01m、位移阈值 0.05m、角度 15°。手部控制参数和 mesh 资产仍须在正式实验清单中固定；baseline 分支与 main 分支结果不可混用。

## 官方与 agent 接口

```bash
python -m dexgrasp_lab reference-plan --profile kpgrasp_fm --python /absolute/kp_env/bin/python --data-root /absolute/dexonomy --output-root /absolute/kp_runs
```

支持 kpgrasp_fm、kpgrasp_flowmap、dexlearn_diffusion、dexlearn_nflow、dexgraspnet2_diffusion。输出 cwd、argv、环境变量、来源 SHA、未解决限制。它们是参考复现计划，不是已完成运行验证的统一训练后端；官方各环境、数据和验证集适配仍需验收后按选定路线落实。

后续 agent 可以按 `doctor → prepare → audit → plan → 人工验收 → train --execute → sample → IK → benchmark` 调度。当前只实现监督式静态抓取生成；如果“agent 训练”指 RL，需要另定义环境、observation/action、reward、终止条件与评测协议，本轮不伪造这些未确定接口。
