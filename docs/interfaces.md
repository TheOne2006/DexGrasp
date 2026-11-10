# Data v1 与命令接口

本版只接受 Sharpa 22 自由度、21 个固定顺序关键点。坐标单位米、关节角单位弧度。

```text
dataset_v1/
├── assets/                       mesh 与原始抓姿 NPZ
├── scenes/<scene_id>.json         实例、canonical object_id、split、视角和抓姿引用
├── hand/hand_spec.json            标定后的手模型；相对路径引用 URDF
├── shards/part-00000.h5           prepare-data 生成
├── splits/{train,val,test}.jsonl  prepare-data 生成
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
- `approach_axis=[0,0,1]`；`reference_length_m` 固定手部参考长度，与坐标尺度独立。
- `palm.normal` 为单位测量法向；`palm.ray_origins_m[N,3]` 是预先标定并均匀覆盖固定平面区域的射线起点，N≥64。
- `keypoint_map_version`、`label_version` 冻结后用于跟踪定义变化。

FK 得到 `K=R FK(q)+t`，`d=R[:,2]`。掌面射线在完整目标 mesh 上找第一非负交点，至少命中 64 条，h 为命中距离中位数。目标 mesh 须闭合且面朝向一致，以检测掌面射线起点是否已进入物体；未命中或穿透记录无效原因，不截成零。

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

## FM、FlowMap 与恢复

`python -m dexgrasp train --config configs/{fm,flowmap}.json` 默认仅检查计划，`--execute` 才更新参数。FlowMap 在线生成训练集条件下的教师轨迹，抽同一轨迹的 `(K_a,K_b)`，不缓存或混用 val/test 蒸馏监督。教师冻结且处于 eval 模式。

检查点包含配置、模型/优化器状态、源码指纹、划分 hash、HDF5 文件尺寸/mtime、手模型和教师指纹、依赖版本/原生扩展指纹、RNG 与步数。HDF5 尺寸/mtime 是变化检测，不是完整内容 hash；文件应冻结并保留数据版本。恢复检查同一配置、代码和数据身份；prefetch 不影响 batch 游标。改变实验配置时使用新 run。

验证只用于诊断：FM 用固定随机数的速度 MSE；FlowMap 用固定噪声的生成终点与配对真值 MSE。多模态任务的配对 MSE 不等同于生成质量，更不是抓取成功率。`best.pt` 仅按该诊断指标选取。

## test 与物理结果

`sample` 生成米制 K，保存 d/h 和样本身份；`decode` 输出每个候选的 R/t/q、重建误差、方向误差和重新测量的掌面间距。IK 使用解析 FK 雅可比和真实 URDF 关节限位，不做无报告的角度截断。不可达候选会标记失败。

`test --decoded decoded.jsonl` 报告几何控制结果，未测物理结果为 null。外部评估器可通过 `--verdicts verdicts.jsonl` 提供逐候选结果：

```json
{"sample_index":0,"candidate_index":0,"collision_free":true,"physical_success":false,"evaluator":"实际评估器名称","protocol_version":"已冻结版本","decoded_sha256":"该 decoded.jsonl 文件的 SHA256"}
```

索引对应同一次 sample/decode 文件；接口强制核对 decoded 文件 SHA256，拒绝混用其他候选的结果。缺少任何候选的物理结果时，只报告覆盖率与已验证成功数，完整联合成功率保持 null。全部结果齐全时，联合成功需同时满足 IK/限位、间距容差、方向容差、无碰撞和物理成功；测量无效与无解留在分母中。

CLI 默认间距容差 5 mm、方向容差 10°、IK 重建容差 3 mm 只是工程初值，正式 test 前应按任务冻结。sample 计时含点云编码与生成，decode 单独计时；没有外部碰撞、仿真和规划计时就不能声称端到端耗时。

## 新场景与条件扫描

`sample --palm-gap-m 0.02 --direction 0 0 1` 只覆盖推理条件，不修改训练标签。固定 `--seed` 和场景、方向，改变间距可做连续控制扫描；固定条件改变 seed 可看候选多样性。样本文件记录实际使用的 h/d 与覆盖参数。

没有训练标签的新场景可直接调用 `dexgrasp.test.inference.predict(model, raw_condition, candidates=8)`。raw_condition 使用批量米制点云与 h、场景系方向；函数统一归一化并返回米制 `[B,8,21,3]`。模型由 `dexgrasp.train.runner.load_model` 加载。此 API 不要求新场景出现在训练 HDF5 中，几何解码与物理验收仍独立进行。
