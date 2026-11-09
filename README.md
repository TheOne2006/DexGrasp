# DexGrasp

EPIC 灵巧手抓取姿态生成研究代码。当前提供多架构实验准备框架；数据集、最终模型与研究贡献尚未确定，尚未启动训练。

本目录是自有 Git/GitHub 代码仓库；上级目录是研究工作区。参考代码放在上级 `reference/`，按需要使用，见 [项目结构](docs/project-structure.md) 和 [参考代码使用说明](docs/reference-code.md)。项目背景与论文阅读资料留在外层工作区。

数据、权重与运行结果使用上级目录的 `data/`、`checkpoints/`、`outputs/`，不提交大文件。实验前记录代码版本、配置、数据划分和评价协议，区分关键点/IK 拟合成功与物理抓取成功。

## 三部分

- data：`dexgrasp_lab/data.py`、`prepare.py`，统一 manifest、物体级划分、NPZ/HDF5 读取和显式关节顺序 FK。
- train：`models.py`、`methods.py`、`train.py`，PointNet+MLP / PointNet+Transformer × Diffusion / Flow Matching，单卡梯度累积、检查点和恢复。
- test：`test.py`，采样、描述性指标、KPGrasp IK 和 DexGraspBench 命令规划。软件测试独立放在 `tests/`。

这些是自有受控基线，不是 KPGrasp、DexLearn 或 DexGraspNet2 的论文复现。训练只需 PyTorch、NumPy、h5py；不依赖 Isaac Lab。FK/IK/物理评测分开配置环境。

## 先检查，再验收

在本目录使用自己的训练 Python 环境运行：

```bash
python -m dexgrasp_lab doctor
python -m dexgrasp_lab list-models
python -m dexgrasp_lab plan --config configs/pointnet_mlp_flow_matching.json
python -m dexgrasp_lab train --config configs/pointnet_mlp_flow_matching.json
```

最后一条**默认仍只打印计划**。缺少数据时返回 JSON `blocked` 和退出码 2。只有显式添加 `--execute` 才进入训练，本轮没有执行。

配置路径相对于配置文件，manifest 内的数据路径相对于 manifest。训练记录和权重分别写到项目上级的 `outputs/`、`checkpoints/`；拒绝覆盖已有非空运行目录。恢复使用同一配置与 `--resume /absolute/path/last.pt`，并核对 manifest 和代码 SHA256。

详见 [数据格式与命令](docs/interfaces.md)、[项目结构与 GitHub 设置](docs/project-structure.md)、[参考代码索引](configs/references.json)。准备报告位于外层工作区的 `docs/`，不随本代码仓库上传。

模型 test 入口：`python -m dexgrasp_lab test --samples /absolute/samples`。数据划分保持 `train / val / test`。

## 代码检查

FK 检查使用 KPGrasp 的手模型；新机器先运行 `python scripts/setup_references.py --clone KPGrasp`。其余四个 reference 不必为软件检查全部下载。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
```

测试包括模拟中断/恢复；该测试禁用了 `optimizer.step`，不更新模型权重。测试依赖 MuJoCo 以验证 FK。测试夹具均为临时合成数据，不能作为抓取实验结果。
