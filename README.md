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
├── configs/                      FM、FlowMap、未标定手模型模板、参考代码索引
├── tests/                        软件测试；合成夹具不用于研究结论
├── scripts/                      参考代码检查和软件性能检查
├── docs/                         数据、模型、运行和验证说明
├── licenses/                     第三方许可证
└── .github/                      CI
```

实际数据、权重和结果仍分别放在外层 `../data/`、`../checkpoints/`、`../outputs/`；参考源码留在 `../reference/`。详见 [项目结构](docs/project-structure.md)。

## 运行

使用包含 PyTorch、NumPy、h5py、SciPy 的项目环境。真实几何编码还需要与 PyTorch/CUDA 匹配的 **MinkowskiEngine**；缺失时明确报错，不替换成其他模型。当前本机隔离环境与构建方法见 [环境说明](docs/environment.md)。OBJ/STL/PLY 导入额外需要 `trimesh`；NPZ 三角网格不需要它。

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

## 当前边界

- `configs/hand_spec.template.json` 是未标定模板。实际 Sharpa URDF、22 个关节顺序、21 个关键点 link/局部坐标、刚性掌部锚点、掌面射线区域和固定参考长度需要真实数值。
- 当前 Data v1 采用你最新指定的**完整 mesh 表面积均匀采样**；不是可见表面的单视角渲染协议。相机信息保留在场景 JSON，不参与本版采样。
- FlowMap 使用训练 split 条件下冻结 FM 的真实 ODE 轨迹，默认 Heun32，并混入完整 `(0,1)` 区间；学生有独立参数。
- 碰撞与抓稳结果由外部评估器提供。没有外部结果时物理成功率返回 `null`，不会把 IK 成功当作物理成功。参见 [接口](docs/interfaces.md)。
- 已增加完整模型的小数据过拟合实验，使用官方 Sharpa URDF 和明确标注的合成几何诊断标签。初始化、冻结策略、checkpoint 与实验边界见 [过拟合实验](docs/overfit-v1.md)。这不属于真实成功抓取数据训练。

## 验证

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
```

训练循环软件测试禁用 `optimizer.step`。MinkowskiEngine 不可用时，仅真实稀疏网络测试跳过；其他模型测试使用限定在 `tests/` 的编码器替身。本机已通过 40 项测试及完整模型 CUDA 检查，范围见 [验证说明](docs/validation.md) 与 [过拟合实验](docs/overfit-v1.md)，模型与优化细节见 [模型 v1](docs/model-v1.md)。

### Sharpa 1000 条数据与实验展示

基于现有 Dexonomy 物体与 Shadow 抓姿做 Sharpa 显式重定向，保持 22 DOF 与 21 个诊断关键点。数据来源、尺度、筛选、划分和训练方案见 [实验说明](docs/sharpa-1k.md)。[Tailscale 数据浏览网站](http://100.91.234.79:8766) 展示真实 mesh、重定向骨架、检查结果与训练日志。运动学 / 几何检查通过不代表物理抓取成功。

## 预测物理评估与数据生成

新增独立仿真环境和可直接调用的 Sharpa 评估入口，支持 Floating 六向与
Tabletop Slide 后六向协议，以及有界单卡 Init → Refine → Eval / 恢复。
参见 [接口、安装与验证边界](docs/generation-evaluation.md)。
模型 checkpoint 必须匹配训练源码；IK 拟合成功不等于物理抓取成功。
