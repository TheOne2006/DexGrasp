# 抓取数据集 diversity

直接读取 DexGrasp 数据生成的原生 Sharpa 抓姿，输出一个数据集级别的 `Q`。
当前默认：**全局非自身 pair、团队统一标注的 21 点、冻结的 hand50 参数**。
`Q` 在 `[0,1]` 内，越大表示在这一固定特征与尺度下，抓姿越分散。

## 使用

在仓库根目录、已配置好的独立 simulation 环境中执行。首次安装见
[环境说明](../README.md)；已有环境只需更新入口，无需更换依赖：

```bash
python -m pip install --no-deps -e ./simulation

# 全部成功输出的精确值，计算所有 N*(N-1)/2 个 pair
dexgrasp-diversity /path/to/run/checkpoint \
  --asset-root /path/to/assets --mode exact \
  --output ../outputs/diversity/exact.json

# 同一全集，均匀、有放回采样 B=1,000,000 个非自身 pair
dexgrasp-diversity /path/to/run/checkpoint \
  --asset-root /path/to/assets --mode sample --num-pairs 1000000 --seed 0 \
  --output ../outputs/diversity/sample.json
```

`python -m dexonomy.diversity` 与 `dexgrasp-diversity` 等价。只用 CPU，运行时不加载
Torch/Warp、不启动物理仿真。NumPy、MuJoCo、trimesh 已包含在 simulation 环境中。
包括读取与 FK 的计算量：精确模式为 `O(N²)`，采样模式为 `O(N+B)`；内存为 `O(75N + batch_size*75)`，
不创建 `N×N` 矩阵。大数据集可先用采样模式。

- 不给 `--num-pairs` 时，sample 使用 `B = --pairs-per-grasp × N`（默认 `100×N`）。
  exact 忽略采样预算；采样次数允许超过不同 pair 的数量。
- 多个输入路径会合成一个全集。**比较不同模板数时分别运行命令**，每次输入该版本全部输出。
- `--save-features ../outputs/diversity/features.npy` 保存已加权的 `N×75` 特征；
  `--batch-size` 控制 pair 批量，`--feature-batch-size` 控制 FK 批量。
- 输出文件、特征文件须使用新路径，避免覆盖历史结果；结果放在仓库外。

Python 接口使用相同默认配置：

```python
from dexonomy.diversity import evaluate_dataset, evaluate_pair_features

report = evaluate_dataset(
    ["/path/to/run/checkpoint"], asset_root="/path/to/assets",
    mode="sample", num_pairs=1_000_000, seed=0,
)
print(report["dataset_diversity"])

# 已保存的加权特征也可直接计算；不要再次施加权重
# report = evaluate_pair_features(z, mode="exact")
```

## 输入与筛选

接受原生单条抓姿 `.npy`、抓姿目录，或包含 `success/` 子树的 checkpoint/run 目录。
发现 `success/` 时只读取该子树，避免重复读取 raw/refine、轨迹及反馈副本。
NPY 内含 Python 字典，只读取可信生成结果；stage ledger、训练 HDF5 和预测 JSONL
不能直接作为此入口输入。

每条数据需包含 `hand_name=sharpa_v2`、`grasp_qpos` 和 `scene_path`。
`grasp_qpos` 为 `(29,)` 或 `(1,29)`，按 `xyz[3], quaternion_wxyz[4], q[22]` 排列，
长度为米、关节角为弧度，关节顺序与生成器一致。读取实际 grasp，不能用 pregrasp、
squeeze 或末端控制目标替换。`obj_pose`/`obj_scale` 如提供，须与初始 scene 一致。

`--asset-root` 下需要 `hand/sharpa_v2/right_kp.xml` 及其引用文件，并能解析
`scene_path` 和目标物体 mesh；支持资产目录迁移后的路径解析。21 点定义已随包提供，
无需另行提供整份 `sharpa_21point_v1`。当前支持单个 rigid object 目标的
Floating/Tabletop 输出，`--protocol auto` 自动识别。

默认 `--selection success` 读取成功判定：Floating 的 `stability_metrics.passes`、
Tabletop 的 `tabletop_metrics.joint_success`；无判定时只接受 `success/` 下的记录。
显式失败会被排除。若要分析失败抓姿，须显式输入失败文件/目录并用 `--selection all`；
该选项不改变上述目录发现规则。

同一路径只读一次；不同文件重复完整 candidate identity 会报错，防止同一次生成被
重复计数。不同 candidate 的相同姿态保留，作为分布中的重复样本。错误格式、缺失
资产和不一致姿态会报错；应检查结果中的 `counts` 确认实际参与计算的数量。

## 定义与默认特征

对所有入选记录建立统一特征 `z`，令 `U=N(N-1)/2`：

```text
mu = (1/U) * sum_{i<j} exp(-||z_i-z_j||²)
Q  = 1 - mu
```

这里 `exp(-distance²)` 是原始 Gaussian kernel 的平方。所有跨物体、跨尺度 pair
均参与，仅排除同一个记录与自身配对；物体/条件标签只用于来源统计，**不先分组算分再平均**。
采样时用实际抽取的 `B` 替代分母 `U`，估计同一个全集的平均值。

`z` 共 75 维：位置 3 + 朝向矩阵 9 + 手型 `21×3`。物体坐标系下的掌位置以
缩放后 mesh 的表面积加权质心居中，用 `0.5 × mesh RMS 半径` 归一化；
朝向使用物体坐标系下的掌旋转矩阵，尺度为 `sqrt(2) × π/6`；
手型用掌局部坐标系 `right_hand_C_MC` 下的关键点，尺度 `0.02 m`，21 点等权 `1/21`。
每个特征块再乘其权重的平方根：

| 位置 `w_t` | 朝向 `w_R` | 手型 `w_h` |
| --- | --- | --- |
| 0.0026744040293024793 | 0.022772101424186664 | 0.23440514780267566 |

关键点依次为 wrist，以及 thumb/index/middle/ring/pinky 各 4 点，指尖由 URDF 固定链
映射到 MJCF；5 个刚性锚点仍计入等权定义。资源、关节顺序及来源 SHA256 见
[`sharpa21_v1.json`](../dexonomy/diversity/resources/sharpa21_v1.json)。

**hand50 表示原 20 点参考集上的距离贡献目标为 25%/25%/50%**，不是对每个数据集
强制手型贡献 50%。本版保留这些数值，未按全局 pair 或 21 点重新拟合。冻结配置和
原校准文件 SHA256 见 [`hand50.json`](../dexonomy/diversity/resources/hand50.json)，
评估时不读取外部实验路径，也不根据输入数据重新调参。

## 结果与比较边界

JSON 的主要字段：

| 字段 | 含义 |
| --- | --- |
| `dataset_diversity` / `dataset_collision_mean` | 主结果 `Q` / `mu` |
| `total_N` / `total_U` / `total_B` | 记录数 / 全集不同无序 pair 数 / 实际计算 pair 数 |
| `scores.joint` | 联合特征的分数与 pair 采样误差 |
| `scores.translation/rotation/hand` | 单特征块的诊断分数，不能相加当作联合分数 |
| `hand` / `kernel_profile` / `feature_config` | 点定义、固定配置及来源指纹 |
| `counts` / `pair_population` | 筛选计数与全局 pair 构成 |

全集不足 2 条时分数为 `null`，状态为 `no_nonself_pairs`。
采样的 `Q_mc_95_interval` 使用 Hoeffding 界，`pair_mc_standard_error` 为均值标准误；
它们只描述**固定全集下抽 pair** 的误差，不包含抓姿生成或子集选择的不确定性。

比较时固定同一关键点定义、尺度、权重和筛选规则。均匀随机子集的精确 pair 均值
对全集均值无偏，但仍有子集波动；不同样本量本身不再通过 `1/N` 对角项影响评分。
改变物体占比、重复次数或生成分布仍会改变 `Q`，记录多的物体自然占据更多 pair。
跨物体直接比较原生 mesh 坐标轴，没有自动语义对齐。高 `Q` 不代表抓稳率或物理质量高。

旧 20 点结果与本版 21 点结果不可直接混比。此前三个固定 1,000 条子集改为 21 点后，
联合 `Q` 下降约 `0.0391–0.0393`；这一差异来自表示变化，并未通过重拟合参数消除。

## 软件验证

在仓库根目录运行：

```bash
PYTHONPATH=simulation CUDA_VISIBLE_DEVICES='' \
  python -m unittest discover -s simulation/tests -p 'test_diversity*.py' -v
```

基础测试使用临时原生数据和独立 pair 公式；设置 `DEXGRASP_ASSET_ROOT=/path/to/assets`
可额外运行真实 Sharpa FK/原生入口检查。CI 不依赖私有数据或模型资产。
