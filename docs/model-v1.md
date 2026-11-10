# 自有条件关键点模型 v1

这是用户模型设计的实现，不是 KPGrasp 论文实验复现，也不是全部 Flow Map Matching 目标的实现。

## 主干

目标分支 `[B,1024,3]` 与全 1 标记，场景分支 `[B,2048,3]` 与目标 1/环境 0 标记分别进入独立 MinkUNet14。两路均保留原稀疏网络结构，使用 1 通道输入特征、512 通道输出；TensorField 负责体素化及恢复原点顺序，混合目标/环境体素取输入特征均值。坐标尺度为 L 时，网络内体素尺寸使用 `voxel_size_m/L`。

逐点特征经 FPS128、Linear(C_A,512)，加 PositionMLP(3→128→512)。关键点 K 经 3→128→512 与 21 个 keypoint ID 相加。每个 block 依次做：

1. AdaLN + 8×64 self-attention + gated residual。
2. AdaLN + 目标 cross-attention + 方向偏置 + gated residual。
3. AdaLN + 场景 cross-attention + 方向偏置 + gated residual。
4. AdaLN + 512→2048→512 GELU FFN + gated residual。

共 8 层，参数不共享，dropout=0。各子层 modulation 独立 SiLU→Linear(512,1536)，输出 γ、β、gate。全局条件为 `MLP(h/L_hand)+time_embedding`；FM 单时间用 Sin128→512→512，FlowMap 两个 Sin128 拼接后 256→512→512。

方向 d 在外部输入处归一化且零向量报错。以 r=p−x、s=r·d、ρ=‖r−sd‖ 构造 `[B,J,128,2]`，每层每分支独立 2→64→8 MLP，加入 attention logits。方向不进入 AdaLN，h 不进入几何偏置。几何偏置不区分绕轴方位；绕轴旋转、位置和手指形状由内容特征与生成状态共同决定。

## 两种生成方法

FM：`Kτ=(1−τ)K0+τK1`，速度目标 `K1−K0`，均方误差训练。输出 LN→Linear512→3，默认 10 步 Euler 推理。

FlowMap：冻结完整 FM（包括几何编码器），在训练集条件 C 下从 K0 实际积分，抽取同一条轨迹的 a<b；学生输出 `Ka+(b−a)u(Ka,a,b,C)`，直接对教师 Kb 做端点 MSE。默认 Heun32 教师轨迹、完整 `(0,1)` 占比 0.5，其余为均匀离散时间网格上不同端点对。两种网络独立实例化，学生不共享或更新教师参数。不除以时间差；a=b 外层返回原输入。

FlowMap 一步调用 `(0,1)`，多步连续调用区间映射；不会再次对映射输出做 Euler 更新。默认配置的采样步数为 1，可在 sample CLI 指定更多步。

## 效率措施与边界

- 一次采样只编码一次点云；同一条件的多个候选共享初次编码后再扩展上下文。
- 推理缓存每层两路 cross-attention 的 LayerNorm 与 K/V 投影；缓存仅限 eval/no_grad、同模型和同一组条件，训练不使用跨步缓存。
- 每次 forward 只计算一次两路 `(s,ρ)`；每层独立 MLP 仍保留。下一次生成调用重算关系。
- 使用 PyTorch SDPA，自动选择支持可学习加性偏置的实现；不保证一定启用 FlashAttention。
- 可选 BF16 用于 Transformer；MinkUNet、几何关系和轨迹状态/损失保持 FP32。初版配置默认 FP32。
- HDF5 按行读取，观测缓存与文件缓存有界；DataLoader 支持 persistent workers、CUDA pinned memory、nonblocking transfer。
- IK 提供解析 FK Jacobian，减少数值差分的反复 FK。

缓存等价性、SDPA 与显式 softmax 的一致性、FK Jacobian 与有限差分的一致性都有软件测试。性能检查只报告实际测量范围；网络调用减少不能直接推算物理规划流水线的提速。

## 未保证的性质

速度/映射 MSE 没有额外添加间距回归、接触或碰撞损失。连续 h 条件不自动保证单调控制，方向 logits 偏置不保证方向精确相等，生成的关键点不自动满足刚性与运动学可达性。训练数据、几何解码和独立验收共同决定有效性。

采用的来源及许可证见 `THIRD_PARTY_NOTICES.md`。TensorField 恢复原点顺序的接口依据 [MinkowskiEngine 官方文档](https://nvidia.github.io/MinkowskiEngine/sparse_tensor.html)；SDPA 使用 [PyTorch 官方接口](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。
