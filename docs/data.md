# Data v1：数据格式与来源

本版只接受 Sharpa 22 自由度、21 个固定顺序关键点。坐标单位米、关节角单位弧度。

```text
dataset_v1/
├── assets/                       mesh 与原始抓姿 NPZ
├── scenes/<scene_id>.json         实例、canonical object_id、split、视角和抓姿引用
├── hand/hand_spec.json            标定后的手模型；相对路径引用 URDF
├── shards/part-00000.h5           prepare-data 生成
├── splits/{train,test}.jsonl  prepare-data 生成
├── preparation.json              输入指纹、采样协议和样本数
└── invalid_labels.jsonl           无效抓姿/间距的原因
```

## 原始场景

每个 JSON 对应一份观测；`object_id` 是跨尺度、视角和抓姿一致的物体身份，不能用实例 ID 代替。不同观测可使用不同文件名，但 `(scene_id,view_id)` 必须唯一。同一场景与同一物体都不能跨 split。相机信息可记录为对象或 `null`。

```json
{
  "scene_id": "S001", "view_id": "V003", "object_id": "bottle_asset_001",
  "split": "train", "target_id": "bottle", "camera": null,
  "collision_validation_version": "填写真实外部验证协议版本",
  "grasps": "assets/S001_grasps.npz",
  "instances": [
    {"instance_id": "bottle", "mesh": "assets/bottle/mesh.npz", "scale": 1.0,
     "R": [[1,0,0],[0,1,0],[0,0,1]], "t": [0,0,0]},
    {"instance_id": "table", "mesh": "assets/table/mesh.npz", "scale": [1,1,1],
     "R": [[1,0,0],[0,1,0],[0,0,1]], "t": [0,0,0]}
  ]
}
```

mesh NPZ 字段为 `vertices[N,3]` 和整数 `faces[F,3]`；其他网格格式通过可选 trimesh 导入。先按实例 scale 与 R/t 变换到场景系，再按变换后的面积采样。环境各实例合并后按总面积采样，不能给每个物体相同点数。目标独立采 1024；场景重新采目标 512、环境/桌面 1536；场景点和 mask 同步重排。

抓姿 NPZ 包含 `R[G,3,3]、t[G,3]、q[G,22]、valid[G] bool、grasp_id[G]`。`valid` 是已有外部抓姿有效性检查的结果，prepare 不会自行模拟抓取。放入新障碍后的旧 success 标签不能直接沿用。

## 手模型与标签

从 `configs/hand_spec.template.json` 填写真实数据，不得仅把 `calibrated` 改为 true：

- `urdf`、`root_link`、`joint_order[22]`；适配器支持有界 revolute/prismatic 与 fixed，遇到 mimic/continuous 等未支持关节明确拒绝。需要扩展时，依据真实 Sharpa 资产验证关节定义。
- `keypoints[21]` 各含 `name、link、local_xyz_m、palm_anchor`；至少 3 个不共线锚点必须对 root 刚性固定。
- `approach_axis=[0,0,1]` 仅保留旧数据兼容；当前条件由 `direction_condition={kind: official_wrist_positive_z_v2, local_axis: [0,1,0]}` 明确规定。`official_from_grasp_rotation` 必须与实际 URDF 固定变换一致；`reference_length_m` 固定手部参考长度，与坐标尺度独立。
- `palm.normal` 为单位测量法向；`palm.ray_origins_m[N,3]` 是预先标定并均匀覆盖固定平面区域的射线起点，N≥64。
- `keypoint_map_version`、`label_version` 冻结后用于跟踪定义变化。

FK 得到 `K=R FK(q)+t`，`d=R[:,1]`（当前 wrist-z-v2：官方 +Zh = 抓取坐标 +Yg；旧 v1 的 `R[:,2]` 仅供历史复现）。掌面射线在完整目标 mesh 上找第一非负交点，至少命中 64 条，h 为命中距离中位数。目标 mesh 须闭合且面朝向一致，以检测掌面射线起点是否已进入物体；未命中或穿透记录无效原因，不截成零。

这只是掌面标签有效性检查，不是整只手的碰撞检测。测量区域适用性仍需先在瓶身侧抓子集标定。

## HDF5

| group | 字段 | dtype / 每行形状 |
| --- | --- | --- |
| observations | object_points / scene_points | float32 `[1024,3]` / `[2048,3]` |
| observations | scene_mask | uint8 `[2048]`，恰好 512 个 1 |
| observations | scene_id / view_id / object_id | UTF-8 string |
| grasps | observation_index | int64 scalar，引用本 shard 观测 |
| grasps | grasp_id | UTF-8 string |
| grasps | R / t / q / K / d / h | float32 `[3,3] / [3] / [22] / [21,3] / [3] / [1]` |
| grasps | hit_ratio | float32 scalar |

文件 attributes：`schema_version=1`、`hand_spec_sha256`、`label_version`、`point_sampling=full_mesh_area_uniform`。分片阈值可能使同一观测出现在相邻 shard；每个 shard 内的多条抓姿只引用一份点云。

split 每行：`{"shard":"shards/part-00000.h5","grasp_index":0}`。路径相对于 dataset 根，拒绝路径越界、负索引、重复引用和跨 split 泄漏。审计检查全部引用行的 dtype、形状、有限性、R、d、h、命中数、关节限位及 K 与标定 FK 的一致性。

## DataLoader

`GraspDataset` 输出 `condition`、`keypoints_gt`、`metadata`。训练和推理共用同一归一化：目标点均值为 o，固定尺度 L，object/scene/K 都执行 `(x-o)/L`；d 不缩放，h 保持米制，由模型除以手参考长度。

HDF5 文件句柄和观测 LRU 缓存各 worker 独立，均有容量上限。只读取指定抓姿行，没有把整个 shard 载入内存。FM 噪声与时间在训练时生成，不保存为静态样本。

## 现有 Sharpa 1000 条诊断数据

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

从确定性打乱后的候选物体中，已选出前 20 个各有 50 条通过的物体，形成 800/200 的 train/test（原 100 条 val 已并入 test），物体级无交叉。去重依据原始 root/joints 与尺度的精确 float32 内容。原始任务为 floating；为了满足模型的环境通道，增加 z=-0.4 m 的远离手与物体的人工地面，仅作为输入上下文，不赋予桌面抓取成功语义。

## 数据位置与验收

数据位于外层 `../data/sharpa_dexonomy_1k_v1/`，生成和审计记录位于 `../outputs/sharpa_dexonomy_1k/`；这些路径从仓库根目录解析。

最终恰好 1000 条，20 个物体，800 / 100 / 100。源候选池 81,076 条；每个候选物体获得 50 条通过样本即停止，所以并未对全池逐条完成 IK。共收集 3,899 条合格候选，从中按预定物体顺序取 1000 条。重定向 MPJPE 中位数 6.398 mm、P95 7.499 mm。所有 1000 条均通过独立 MuJoCo FK 复核，最大误差 9.463e-9 m。20 个源 OBJ 的精确字节哈希不同；此检查不涵盖近似重复几何。

早期 12 条合成几何样本仅用于训练链路诊断，不是真实成功抓取标签。现有 Sharpa 重定向标签也只经过运动学和有限几何检查，尚无完整手部碰撞、自碰撞或物理抓取成功率验证。

## 数据准备与审计

在仓库根目录执行：

```bash
python -m dexgrasp prepare-data --root ../data/dataset_v1
python -m dexgrasp audit-data --root ../data/dataset_v1
```

模型输入、条件注入与生成架构见 [model v1](model-v1.md)。数据展示板独立位于外层 `../site/`，启动方式见 [site README](../../site/README.md)。

## +Zh 方向标签修正（2026-11-10）

用户明确选择截图中的官方 +Zh，正向为手腕朝手指，不是物体中心到手根的方位。保留 URDF/FK 的抓取坐标 g；`R_h=R_g C.T`，因此 `d=R_h[:,2]=R_g[:,1]`。掌物间距仍沿 `R_g @ palm.normal = R_g[:,2]` 测量。方向与位置无关，不能据此推断机械臂所在侧或可达性。

新数据 `sharpa_dexonomy_1k_wrist_z_v2` 独立复制旧数据并仅重写 d 与版本元数据；800 train / 200 test、点云、R/t/q/K/h 全部保持不变。`direction_migration.json` 包含全量审计与旧新身份。旧 checkpoint 不接受新标签身份。

本轮从随机初始化训练 FM，8 卡 DDP，每卡 256、全局 2048、累积 1，10,000 optimizer steps，AdamW 学习率 2e-4；每 1000 step 永久保存，最终 step 10000 独立测试，不用 test 选模型。

## 2026-11-11 正式 object-only 数据

全量 surface-meanh（4,168,807）加独立 tabletop 增量（159,922），总计 4,328,729；首批子集不重复加入。
正式打包入口是 `scripts/pack_object_corpus.py --source <source> --output <new-local-root> --formal --seed 42`。
逐文件 SHA-256 校验后，全量核验唯一 sample ID、有限值、关节限位、旋转、方向与 URDF FK。
本次 50,407 个源文件通过，4,328,729 行 FK 最大绝对误差 3.045e-8 m。

划分将 object_id 与完全相同的 source_mesh_sha256 合并；4096 个 object_id 对应 4091 个资产组。
对组名使用 SHA-256(`dexgrasp-formal-v1:42:<group>`) 前 16 个十六进制位 mod 10，等于 0 为 test。
最终 train=3,899,610、test=429,119；四组共用冻结数组及 split_protocol.json。
parent 是跨物体复用的生成/转移模板，不能作为同一物体身份：全量 parent 闭包会合并所有物体。
存在 216,406 个跨物体复用父模板、527,254 条跨划分父子边，已记录在协议。
该划分评估 held-out objects/assets，不声称 held-out generation lineages，也未检查近似网格重复。

源标签包含上游物理协议与 source_success 元数据，但当前新模型没有物理抓取成功率结果。
本轮 h-less 模型不读 h，不将新数据的 surface-meanh 距离与旧诊断数据的 ray-median 混为一谈。
训练表只存规范化点云、关键点及条件，不存学习特征；正式 runner 启动验证全部 packed 数组摘要与冻结划分。
