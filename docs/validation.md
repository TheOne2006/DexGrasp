# v1 软件验证

验证使用合成几何与抓姿夹具，训练循环的 `optimizer.step` 被禁用；没有正式训练、真实抓取成功率或模型质量结论。验证原始记录放在外层 `../outputs/v1_verification/`，不进入 Git。

最终项目环境检查：**38 项测试全部通过，无跳过**，包含真实 MinkUNet CPU 前反向。完整尺寸模型也已通过 RTX 5090 的真实 CUDA 双编码器与 FM/FlowMap 前反向、采样检查。

## 验证覆盖

- 数据：面积采样分布、观测复用、统一归一化、worker 独立句柄、有界缓存、严格 dtype/shape、索引越界、物体/场景 split 泄漏、手模型指纹、FK 标签一致性、原始场景到分片的制作。
- 几何：第一交点、中位数、64 命中阈值、穿透无效、刚体变换不变性、FK 解析 Jacobian、SVD/IK 往返、反射修正、不可达姿态、限位浮点舍入。
- 网络：FM 端点、所有分支梯度、SDPA 与显式 softmax、方向轴向/径向关系、K/V 缓存等价性、候选共享编码、FlowMap 恒等映射、多步映射语义、同轨迹抽对、冻结教师、BF16 反向传播、新场景推理。
- 流程：FM→教师→FlowMap、确定性恢复、计划模式、条件覆盖不修改数据、采样→解码→评估、外部物理结果文件身份、未测物理结果不计成功。

## 已保存的检查记录

`unit-tests.log` 是项目隔离环境的完整软件测试记录；无 MinkowskiEngine 的独立检查另存 `unit-tests-without-mink.log`，其中真实稀疏后端测试会显式跳过。

`cpu-minkunet.json` 已验证真实 MinkUNet14 双分支与较小 Transformer 的 FM/FlowMap 前反向及采样，所有参数梯度有限；稀疏 backbone 本身未缩小。`cuda-transformer.json` 已验证完整 512 维、8 层、128 token 的 Transformer，几何编码使用仅测试用替身。

首次 RTX 5090 Transformer 检查中，B=2、FM10、FP32、三次计时中位数：不缓存 cross-attention K/V 约 110 ms，缓存后约 97 ms，约 1.13×；同噪声输出最大绝对差 0。该范围不包含真实 MinkUNet、SVD/IK、碰撞、仿真或规划，不能解读为端到端加速。CPU 小 Transformer＋真实 MinkUNet 的同项提升约 1.01×，说明实际收益取决于占比。

## 完整模型 CUDA 实测

CUDA 原生扩展完成兼容修补后，`cuda-full-model.json` 使用真实双 MinkUNet14、512 维/8 层/8 heads/128 tokens，B=2、FP32、三次测量中位数：

| 检查 | 编码＋生成时间 |
| --- | --- |
| FM10，不缓存 cross-attention K/V | 161.97 ms |
| FM10，缓存 cross-attention K/V | 122.61 ms |
| FlowMap1，缓存 cross-attention K/V | 53.13 ms |

同噪声的缓存/不缓存输出最大绝对误差为 0；FM 缓存加速约 **1.32×**（耗时下降约 24%）。FM 参数量 91,662,659；FlowMap 参数量 91,728,195。两个模型均通过所有参数梯度有限性检查，优化器更新数为 0。另用同一组权重比较真实 MinkUNet＋较小 Transformer 的 CPU/CUDA 前向，最大绝对误差为 4.77e-7，记录见 `cpu-cuda-agreement.json`。

上述数据是随机初始化、合成输入的工程耗时，不能说明一步学生保留了教师的真实抓取质量。计时包含真实点云编码，但不包含 SVD/IK、碰撞、物理仿真和规划。

当前环境构建来源与复现方法见 [环境说明](environment.md)，最终源码与原生扩展身份保存在外层验证报告中。

## 复查命令

```bash
OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m unittest discover -s tests -v

# 真实稀疏骨干检查；small 仅缩小 Transformer 与输出通道。
OMP_NUM_THREADS=4 .venv/bin/python scripts/check_performance.py --device cpu --small \
  --output ../outputs/v1_verification/new-cpu-check.json

# 完整模型 CUDA 检查（需要 CUDA-enabled MinkowskiEngine）。
.venv/bin/python scripts/check_performance.py --device cuda:0 \
  --output ../outputs/v1_verification/new-full-model-check.json

# 完整尺寸主干单独检查。此开关明确排除稀疏骨干，不是生产模型选项。
.venv/bin/python scripts/check_performance.py --device cuda:0 --transformer-only \
  --output ../outputs/v1_verification/new-transformer-check.json
```

性能脚本拒绝覆盖结果文件，记录模型尺寸、设备、源码指纹、梯度与优化器更新数。代码、原生扩展或模型配置有改动后应重新运行对应检查。

## 仍需真实材料验证

正式训练需要真实 Sharpa URDF、21 点/22 关节映射、掌面区域、固定参考长度，以及有效 Sharpa 抓姿和完整场景资产。本地现有上游预览不是此协议的已标定训练集。模型的掌面间距单调控制、方向误差、避障、物理成功与多样性均需在真实数据和冻结评估协议下验证。
