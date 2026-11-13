# 自有条件关键点模型 v1

这是用户模型设计的实现，不是 KPGrasp 论文实验复现，也不是全部 Flow Map Matching 目标的实现。

## 对照实验版本

通过配置的 `model.variant` 选择版本；未填写的旧配置和旧 checkpoint 按 `full` 解释。
FM 与 FlowMap 共用下面的结构开关，蒸馏时教师与学生要求版本一致。

| variant | 目标 cross-attention | 场景编码器及 cross-attention | 方向条件 | palm_gap AdaLN |
|---|---|---|---|---|
| `full` | 保留 | 保留 | 两路方向几何偏置 | 保留 |
| `full_h_less` | 保留 | 保留 | 两路方向几何偏置 | 移除 |
| `object_only` | 保留 | 移除 | 目标分支方向几何偏置 | 保留 |
| `object_only_h_less` | 保留 | 移除 | 目标分支方向几何偏置 | 移除 |
| `object_direction_adaln` | 保留 | 移除 | 方向 MLP → AdaLN | 保留 |
| `object_direction_adaln_h_less` | 保留 | 移除 | 方向 MLP → AdaLN | 移除 |

方向 AdaLN 使用归一化的 `grasp_direction_cmd[B,3]`，经 Linear(3,128) → SiLU → Linear(128,width)
得到 `e_direction`，加入时间及可选的 gap embedding，调制所有保留子层的缩放、偏移和门控。
这两个版本不计算 `(s,ρ)`，不创建方向几何偏置 MLP；目标点云 token 和位置编码仍保留。
`h_less` 不创建 gap MLP、不读取 `palm_gap`，不是给 gap MLP 输入零。
其余版本的 AdaLN 条件分别为 `e_time + e_gap` 或 `e_time`。

模型与原始输入推理接口仅要求当前版本使用的条件；额外的被消融条件会被忽略。
训练 Dataset 按版本返回条件，DDP GPU 缓存及 FPS 也只处理保留的输入。
底层 HDF5 数据协议和完整性审计保持一致，因此所有版本使用相同的有效样本与 train/test 划分。
生成结果记录 `model_variant` 和 `palm_gap_conditioned`；`h_less` 的样本文件仍保留 h 标签用于
共同评价协议下的参考误差，它不是送给模型的控制命令。现有 gap 误差和几何通过率评价口径不变。

`configs/ablations/` 提供六份 Sharpa wrist-z-v2 FM 对照配置：同一数据集、seed 42、
8×256 全局 batch 2048、10,000 optimizer steps、BF16、随机初始化；每个版本有独立输出及 checkpoint 目录。
这里的八卡 batch 解释适用于 `scripts/train_fm_ddp.py`；普通单进程入口使用本地 batch 256。
配置只是实验准备，不表示已经训练或取得结果。结构移除后参数量不同，固定 seed 不保证共有模块初值逐元素相同。

`full` 保持原有参数名称、张量形状及计算路径，可严格加载旧 full 权重；跨版本权重初始化会被拒绝。
原有严格断点恢复仍要求源码、运行配置等身份完全一致，不因旧权重可加载而放宽。

## 主干（full）

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

采用的来源及许可证见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。TensorField 恢复原点顺序的接口依据 [MinkowskiEngine 官方文档](https://nvidia.github.io/MinkowskiEngine/sparse_tensor.html)；SDPA 使用 [PyTorch 官方接口](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。

## 原始输入的架构说明（原文）

以下原样保留“模型设计”聊天中首次粘贴的模型架构部分，原文中的 `[图片]` 是附件占位符，未包含图片。原始附件后续的 Data v1 部分由 [data 说明](data.md) 中的现行数据协议承接。

原文描述单视角可见表面点云；后续 Data v1 输入已改为完整 mesh 表面积均匀采样，当前实现采用后者。原文中的拟定配置和待标定项目保留原意，当前实现细节以上文为准。

```text
模型架构 v1
1. 模型任务与最终输出
模型接收目标物体、周围障碍物／桌面、连续掌面间距 palm_gap，以及必需的抓取方向，生成满足条件的抓姿。palm_gap 表示瓶身靠掌表面到掌面的间距，越小通常越靠掌；它控制靠掌程度，不单独规定手指接触分布。方向命令规定官方手腕 +Zh 的朝向（手腕朝手指）；手根位置与绕受控轴的旋转由模型结合场景生成。
[图片]
采用同一套条件关键点架构：先训练 FM 生成器，再将其轨迹蒸馏为 FlowMap 少步生成器。二者均先生成整只手的关键点 $$K$$，再通过 SVD＋IK 输出手根朝向 $$R$$、位置 $$t$$ 与关节角 $$q$$。
统一任务：场景、连续掌面间距与抓取方向条件 → 手部关键点 → 可执行抓姿。
$$K\sim p_{\theta}(K\mid P_{\mathrm{obj}},P_{\mathrm{scene}},M_{\mathrm{scene}},h,d_{\mathrm{cmd}})$$
$$K\xrightarrow{\mathrm{SVD+IK}}(R,t,q)$$
- $$P_{\mathrm{obj}}$$：独立采样的目标点云；$$P_{\mathrm{scene}}$$：完整场景点云（目标 512 点＋周围物体及桌面 1536 点）；$$M_{\mathrm{scene}}$$：逐点目标 0/1 标记。$$F_{\mathrm{obj}},F_{\mathrm{scene}}$$ 为编码后的几何特征；加入对应位置编码后得到 $$G_{\mathrm{obj}},G_{\mathrm{scene}}$$。scene 明确包含目标与环境。
- $$K=\{k_j\}_{j=1}^{J}$$ 是手部关键点，$$q\in\mathbb R^{22}$$ 是关节角。手根为手模型固定的 wrist/root link：其局部坐标系记为 H，场景坐标系记为 S。当前存储的 H 是项目抓取坐标 g（diagnostic_grasp_frame），不是官方手根坐标 h；两者同原点、固定换轴，+Yg=+Zh。$$R\in SO(3)$$ 表示 H 相对 S 的朝向，三列分别是手根 x、y、z 轴在场景系中的方向；$$t\in\mathbb R^3$$ 是手根原点在场景系中的位置。局部点与场景点满足 $$p^S=Rp^H+t$$。改变 R 会转动整只手，改变 t 会平移整只手，改变 q 则改变手指相对手根的弯曲。
- $$h=\mathrm{palm\_gap}$$ 是以米为单位的连续掌面间距，越小表示物体越靠掌。输入 $$d_{\mathrm{cmd}}\in\mathbb R^3$$ 是场景坐标系中的手腕 +Zh 单位方向，3 维表示、2 个独立自由度。输出 $$R,t$$ 仍为完整手根位姿，其中位置与绕受控轴的旋转由模型决定。
2. 模型全部输入
每次训练只使用一个确定手型。采用 Sharpa，关节自由度 $$D=22$$；关键点数 $$J$$ 由该手型的固定关键点映射确定。
B：batch size，一次送入模型的样本数；D：关节自由度；J：手关键点数
[B,N,3]：B 个样本，每个样本 N 个点，每点为 (x,y,z) 三个坐标
目标分支独立采样 1024 点；场景分支重新采样 2048 点＝目标512＋周围物体及桌面1536
scene_mask [B,2048]：目标1／其他0；全部为真实表面采样点
目标 tokens 128；环境 tokens 128；统一 hidden dimension 512
τ：噪声→抓姿的生成时间；不表示物理运动时间
object_points · $$[B,1024,3]$$
语义：目标物体的 1024 个真实表面采样点。
来源与假设：训练与推理均直接使用准确的目标点云和环境点云，实例归属已知。
注入位置：FM 与 FlowMap 的目标几何编码；与环境点共享坐标原点和长度尺度。
scene_points / scene_mask · $$[B,2048,3]\;/\;[B,2048]$$
语义：完整场景的 2048 个真实表面采样点及逐点目标归属标记：目标 512 点，标记为 1；周围物体与桌面共 1536 点，标记为 0。点云与标记逐点对齐，重排时同步调整；0 表示非目标，不表示无效。
来源：从与目标分支相同视角、相同坐标系的原始场景观测独立重新采样。工作区裁剪后，目标表面采样 512 点，周围物体与桌面的可见表面均匀采样共 1536 点，按已知实例归属生成 scene_mask。该采样与目标分支 object_points 的 1024 点采样分开进行；点数不足时从对应真实表面重采样或重复采样，不补零。
注入位置：两字段直接输入 FM 与 FlowMap 的 SceneEncoder；scene_mask 扩展为 $$[B,2048,1]$$，作为逐点输入特征，所有场景点均参与编码和 attention。目标与场景两路使用相同编码器结构，均使用 MinkUNet；不共享权重。下文统一使用 $$F_{\mathrm{scene}}$$、$$G_{\mathrm{scene}}$$ 及场景 cross-attention，表示包含目标与环境的完整场景。
palm_gap · $$[B,1]$$
语义：以米为单位的连续掌面间距 $$h\geq0$$，对应固定掌面测量区域沿手模型掌面法向到目标第一表面的代表距离。较小值表示更靠掌。
来源：训练由完整目标 mesh 与对应真实抓姿离线测量；推理由用户或上游任务指定期望间距。
注入位置：先按固定手部参考长度 $$L_{\mathrm{hand}}$$ 归一化为 $$\tilde h=h/L_{\mathrm{hand}}$$，经 Linear(1,128) → SiLU → Linear(128,512) 得到 $$e_{\mathrm{gap}}$$，与生成时间 embedding 相加后注入每个 block 的 AdaLN。FM 与 FlowMap 使用同一标量定义和尺度；palm_gap 不加入方向几何偏置 MLP。
grasp_direction_cmd · $$[B,3]$$
语义：$$d_{\mathrm{cmd}}$$ 为场景坐标系中的手腕 +Zh 单位方向，满足 $$\|d_{\mathrm{cmd}}\|=1$$，具有 2 个独立自由度。当前受控轴为官方 +Zh，在抓取坐标中为 +Yg，即 `a_hand=[0,1,0]`，与掌面测距法向 +Zg 分开。手模型预先固定局部受控轴 $$a_{\mathrm{hand}}$$，期望输出满足 $$Ra_{\mathrm{hand}}\approx d_{\mathrm{cmd}}$$；绕该轴的旋转、手根位置与手指构型交给模型选择。
来源：训练由真实抓姿旋转生成 $$d_{\mathrm{cmd}}=R_{\mathrm{gt}}a_{\mathrm{hand}}$$；推理由用户或上游任务提供。训练与推理均必须提供方向命令，并将非零向量归一化为单位向量。
注入位置：以当前手关键点与目标／场景 token 中心的相对位置为基础，计算相对于 $$d_{\mathrm{cmd}}$$ 的轴向位移 $$s$$ 与离轴距离 $$\rho$$，由 MLP 输出逐 head 的几何偏置，加入每层两路 cross-attention 的 logits。具体公式见 FM.6，FM 与 FlowMap 使用同一注入定义。
noisy_state / flow_time · $$[B,J,3]\;;\;[B]$$
语义：动态状态均为手部关键点。FM 使用单时间 flow_time [B]；FlowMap 使用起止时间 time_start/time_end，各 [B]。时间表示噪声到抓姿的生成进度。
来源：训练循环采样噪声和时间；推理从高斯噪声初始化。
注入位置：每次采样更新的动态变量；点云、掌面间距与方向命令在整个采样过程中保持固定。
3. 条件关键点生成：FM 教师与 FlowMap 学生
训练关系：先训练条件 FM 教师；教师冻结后，生成同一条件下的 ODE 轨迹，再监督 FlowMap 学生学习有限时间区间的映射。两者复用输入定义、关键点顺序和几何解码，学生持有独立网络参数。
FM 在归一化关键点空间中取噪声 $$K_0\sim\mathcal N(0,I)$$ 与真实抓姿关键点 $$K_1$$，构造 $$K_\tau=(1-\tau)K_0+\tau K_1$$。令固定条件 $$C=(P_{\mathrm{obj}},P_{\mathrm{scene}},M_{\mathrm{scene}},h,d_{\mathrm{cmd}})$$，其中方向命令为必需条件，使用速度回归目标：
$$\mathcal L_{\mathrm{FM}}=\mathbb E\left[\|v_\theta(K_\tau,\tau,C)-(K_1-K_0)\|_F^2\right].$$
推理时积分 $$\mathrm dK_\tau/\mathrm d\tau=v_\theta(K_\tau,\tau,C)$$ 得到终点。FlowMap 则从同一教师积分轨迹取 $$K_a^T,K_b^T$$，直接监督 $$\Phi_\phi(K_a^T,a,b,C)\approx K_b^T$$；一步生成使用 $$\Phi_\phi(K_0,0,1,C)$$。学生学习教师的流映射，训练用的噪声—真值直线插值不等同于教师实际积分轨迹。
比较重点：FM 作为质量基线，FlowMap 检验一步或少步调用能保留多少抓取有效性、掌面间距控制与多样性。两者均输出关键点，均需 SVD＋IK 与碰撞检查；减少网络调用不等于同比例减少端到端时间。
FM：条件 Flow Matching
[图片]
本版本学习瞬时速度场，使用单个生成时间 τ；Conditional Transformer 是骨干结构，Flow Matching 是训练与采样方法。
本模型是一个从三维场景到完整手部关键点的条件生成器。下面按实际 forward 顺序展开：输入层 → 三条条件编码支路 → 关键点嵌入层 → 8 层条件 Transformer → 速度输出层 → 采样与几何解码。所有新增层宽和层数为本方案拟定配置。稀疏卷积 backbone 沿用 KPGrasp 所用 MinkUNet 的原实现，不把未核对的内部卷积参数写成新配置。
FM.0 完整网络层次
KeypointGraspModel
├── 1. InputLayer
│   ├── object_points       [B,1024,3]
│   ├── scene_points        [B,2048,3]，独立重采样512＋1536
│   ├── scene_mask          [B,2048]
│   ├── palm_gap            [B,1]     连续掌面间距，单位米
│   ├── grasp_direction_cmd [B,3]    手腕 +Zh 单位方向（必需，2 个自由度）
│   └── K_τ [B,J,3], τ [B]           当前关键点状态与流时间
├── 2. ConditionBranches
│   ├── ObjectEncoder       → F_obj [B,128,512], xyz_obj
│   ├── SceneEncoder（完整场景+目标标记）     → F_scene [B,128,512], xyz_scene
│   └── GlobalEncoder       → e_global [B,512]（掌面间距＋时间）
├── 3. KeypointEmbedding    → H₀ [B,J,512]
├── 4. KeypointTransformer          → H₈ [B,J,512]
│   └── ConditionalBlock ×8
│       ├── AdaLN + self-attention + gated residual
│       ├── AdaLN + target cross-attention（方向几何偏置）+ gated residual
│       ├── AdaLN + scene cross-attention（方向几何偏置）+ gated residual
│       └── AdaLN + FFN + gated residual
├── 5. VelocityHead         → v_K [B,J,3]
└── 6. SamplerAndDecoder
    ├── FM ODE integration  → K [B,J,3]
    ├── palm rigid registration (SVD) → R [B,3,3], t [B,3]
    └── finger IK           → q [B,D]
FM.1 输入层：建立统一的空间坐标
用可见目标点的中心 $$o$$ 和配置固定长度尺度 $$L$$，共同归一化目标、场景与手关键点。场景不单独居中。方向是单位向量，不做平移或长度缩放；如做旋转增强，点云、关键点与方向同步旋转。原始米制坐标及变换保存在 metadata，输出阶段恢复。
$$\bar p=(p-o)/L,\qquad \bar t=(t-o)/L,\qquad \bar K=(K-o)/L.$$
训练的 $$K_\tau$$ 由真实关键点与高斯噪声插值得到；推理从高斯关键点状态开始。目标、环境、掌面间距与方向命令是固定条件，不参与加噪。
FM.2 条件支路一：ObjectEncoder
输入 object_points [B,1024,3] + 内部全1目标标记 [B,1024,1]
→ 体素化（XYZ 定义稀疏坐标，目标标记作为输入特征；保留点→体素映射）
→ MinkUNet
→ 映射回逐点特征 [B,1024,C_A]
→ FPS128 + gather → [B,128,C_A]，centers [B,128,3]
→ Linear(C_A,512) → F_obj [B,128,512]
→ G_obj = F_obj + PositionMLP(centers)
PositionMLP：3→128→SiLU→512；输出 G_obj、centers
位置编码与几何特征相加，不在 token 维拼接。$$C_A$$ 是原 backbone 的实际输出宽度；它通过一个明确的线性层转成 512，不影响后续层的固定维度。
FM.3 条件支路二：SceneEncoder
输入 scene_points [B,2048,3]：独立重采样目标512＋周围物体／桌面1536
     scene_mask [B,2048]：目标=1，其他=0；编码时扩展末维
→ 与 ObjectEncoder 相同的体素化与输入特征接口
→ MinkUNet（与目标分支同结构）
→ 映射回完整场景逐点特征 [B,2048,C_A]
→ 在完整场景上 FPS128 + gather → [B,128,C_A]
→ Linear(C_A,512) → F_scene [B,128,512]
→ G_scene = F_scene + PositionMLP(centers)
输出 G_scene、centers [B,128,3]
F_scene 为几何特征，G_scene 为加入位置编码后的场景 tokens。
场景 tokens 包含目标与环境的联合几何关系，保留共同坐标系中的位置。取消原 FPS→kNN→PointMLP→两层 attention 的异构障碍编码器。ObjectEncoder 与 SceneEncoder 使用相同的 MinkUNet 结构，但参数独立，不共享权重。
FM.4 条件支路三：GlobalEncoder
连续掌面间距分支
  palm_gap [B,1]，单位米
  → 除以固定 L_hand，得到 h_tilde [B,1]
  → Linear(1,128) → SiLU → Linear(128,512) [B,512] = e_gap

流时间分支
  τ [B] → sinusoidal embedding, width=128
  → Linear(128,512) → SiLU → Linear(512,512) [B,512] = e_time

两条支路逐元素相加
  e_global = e_gap + e_time [B,512]
方向条件通过 FM.6 的几何偏置注入 cross-attention。
$$L_{\mathrm{hand}}$$ 是 Sharpa 手模型配置中固定、可复核的参考长度，以米记录，不随手指姿态、物体或样本改变，也不按每个物体做 min–max 归一化。它与坐标归一化尺度 $$L$$ 分别命名、分别保存。AdaLN 的全局条件仅包含连续掌面间距与生成时间；方向仍通过点对几何偏置注入。连续编码不自动保证控制单调性，需要独立验收。
FM.5 关键点输入嵌入层
K_τ                                      [B,J,3]
→ Linear(3,128) → SiLU → Linear(128,512)   [B,J,512]
→ 加 KeypointIDEmbedding(J,512)           [B,J,512]
→ H₀
每个关键点一个 token；关键点 ID 用来区分掌部锚点、不同指节与指尖。$$J$$ 由手型映射固定，跨样本不能更换顺序。
FM.6 生成主干：ConditionalBlock ×8
8 个 block 结构相同、参数不共享。每个 block 的输入和输出都为 $$[B,J,512]$$。下面四个子层严格依次执行，每个子层完成后更新残差流 $$H$$。
ConditionalBlock(H, G_obj, G_scene, e_global, geom_obj, geom_scene)
│
├── 1. 关键点 self-attention
│   ├── modulation₁(e_global) → γ₁,β₁,g₁，各 [B,512]
│   ├── U = (1+γ₁) ⊙ LayerNorm(H) + β₁
│   ├── Q=Linear512(U), K=Linear512(U), V=Linear512(U)
│   ├── 8 heads × 64 → concat → Linear512
│   └── H ← H + g₁ ⊙ AttentionOutput
│
├── 2. 目标 cross-attention
│   ├── modulation₂ → AdaLN(H) = U
│   ├── Q=Linear512(U)                        [B,J,512]
│   ├── K,V=Linear512(LayerNorm(G_obj))       [B,128,512]
│   ├── b_obj = DirectionBiasMLP_obj(geom_obj) → [B,8,J,128]
│   ├── attn = softmax(QKᵀ/√64 + b_obj)V
│   ├── 8 heads × 64
│   └── H ← H + g₂ ⊙ OutputProjection(attn)
│
├── 3. 场景 cross-attention
│   ├── modulation₃ → AdaLN(H) = U
│   ├── Q=Linear512(U)                        [B,J,512]
│   ├── K,V=Linear512(LayerNorm(G_scene))       [B,128,512]
│   ├── b_scene = DirectionBiasMLP_scene(geom_scene) → [B,8,J,128]
│   ├── attn = softmax(QKᵀ/√64 + b_scene)V
│   ├── 8 heads × 64
│   └── H ← H + g₃ ⊙ OutputProjection(attn)
│
└── 4. FFN
    ├── modulation₄ → AdaLN(H) = U
    ├── Linear(512,2048) → GELU → Linear(2048,512)
    └── H ← H + g₄ ⊙ FFN(U)
每个 modulation head 独立使用 SiLU → Linear(512,1536)，分成缩放、偏移、门控三组 512 维向量，并沿 token 轴广播。掌面间距与时间进入 AdaLN，目标与完整场景分别进入 key/value；方向几何偏置只加到两路 cross-attention 的 logits，不加到关键点 self-attention。第一版 dropout 设为 0。
方向条件几何偏置。在共同归一化坐标系中，令 $$x_i$$ 为本次 forward 输入的当前关键点（FM 用 $$K_\tau$$，FlowMap 用 $$K_a$$），$$p_j$$ 为目标或场景 token 的 FPS 中心，$$d=d_{\mathrm{cmd}}$$ 为手腕 +Zh 单位方向。定义：
$$r_{ij}=p_j-x_i,\qquad s_{ij}=r_{ij}^{\top}d,\qquad \rho_{ij}=\|r_{ij}-s_{ij}d\|_2.$$
$$s$$ 表示沿 +Zh 命令方向的有符号位移，$$\rho$$ 表示离经过当前关键点且平行于命令方向的轴有多远。两路关系张量分别为 $$[B,J,128,2]$$。初版拟定各 block 的目标与场景分支各用一个 Linear(2,64) → SiLU → Linear(64,8)，输出 8 个 head 的偏置，重排为 $$[B,8,J,128]$$；层间及两分支参数不共享，同一 MLP 在点对间共享。MLP 输入仅为 $$(s,\rho)$$，不预设前方权重一定更高。
$$b_{ij}^{(h)}=\mathrm{MLP}(s_{ij},\rho_{ij})_h,\qquad A_{ij}^{(h)}=\operatorname{softmax}_j\left(\frac{(Q_i^{(h)})^\top K_j^{(h)}}{\sqrt{64}}+b_{ij}^{(h)}\right).$$
几何 MLP 学习相对于指定方向的空间偏好；内容注意力利用关键点 ID、物体形状与场景特征区分具体对象。相同 $$(s,\rho)$$ 的点处在同一圆环，几何偏置不区分绕轴方位；具体绕轴旋转由内容特征和生成模型选择。各层只更新隐藏特征，不在层间解码新关键点，因此同次 forward 共用关系张量，下一次采样／映射调用再随关键点更新。
与 Plücker 和 GGA 的关系。过 $$x_i$$ 沿 $$d$$ 的辅助线可写成 $$(d,m_i)$$，其中 $$m_i=x_i\times d$$，于是 $$\rho_{ij}=\|p_j\times d-m_i\|_2$$。实现可直接用相对坐标，无须额外六维线编码网络；保留 $$s$$ 才能区分沿线前后。本方案借鉴 GGA 将几何关系加入 attention logits 的方式，但采用可学习 MLP，并非 EgoX 原版的固定方向余弦偏置；效果需由方向误差、抓取成功率和碰撞率验证。
FM.7 输出头、采样和几何解码
H₈ [B,J,512]
→ LayerNorm(512)
→ Linear(512,3)
→ v_K [B,J,3]

FM 采样循环（初版拟定 10 步 Euler）
  K_{k+1} = K_k + Δτ · v_K(K_k,τ_k,conditions)
  固定 G_obj、G_scene、token 中心、掌面间距与方向；每步重算 time embedding
  由当前 K_k 重算两路 (s,ρ)，各层 MLP 输出方向几何偏置
→ K_final [B,J,3]
→ 还原米制坐标
→ 掌部锚点与手模板刚体配准（SVD） → R,t
→ 在该根位姿下做手指 IK → q
→ 输出 R,t,q、掌面间距误差、方向角误差、IK 状态与重建误差
完整场景输入在每个 block、每个生成步骤都参与预测；后处理仍需用真实手表面检查碰撞。以 $$\arccos(\operatorname{clip}((Ra_{\mathrm{hand}})^\top d_{\mathrm{cmd}},-1,1))$$ 验收方向角误差。几何偏置是学习条件，不保证方向严格相等；不固定输出位置或绕轴旋转。SVD＋IK 是几何解码，不是另一个生成网络。
FlowMap：条件关键点流映射
[图片]
从噪声关键点直接映射到终点，或经少量时间区间到达终点。保留 KPGrasp 风格的关键点表示、目标编码与 SVD＋IK 解码，生成模块改为双时间条件映射网络。本版具体实现为 FM 教师轨迹蒸馏，不将其宣称为已复现 Flow Map Matching 论文中的全部目标。
FlowMap.1 输入与三条条件支路
KeypointMapModel
├── object_points + 全1目标标记 → ObjectEncoder（MinkUNet）
│   → point features → FPS128 → Linear(C_A,512) + PositionMLP → G_obj
├── scene_points [B,2048,3] + scene_mask [B,2048]（重新采样512＋1536）
│   → SceneEncoder（同结构 MinkUNet）
│   → point features → FPS128 → Linear(C_A,512) + PositionMLP → G_scene
├── palm_gap / L_hand → Linear1→128 → SiLU → Linear128→512 = e_gap
├── grasp_direction_cmd + K_a + token centers → geom_obj / geom_scene → 各层方向几何偏置
├── (τ_a,τ_b) → TimePairEncoder = e_pair [B,512]
└── e_map = e_gap + e_pair
TimePairEncoder
  τ_a [B] → Sin128
  τ_b [B] → Sin128
  concat [B,256] → Linear(256,512) → SiLU → Linear(512,512)
  → e_pair [B,512]
训练采样 0 ≤ τ_a < τ_b ≤ 1；另包含完整区间 (0,1)。
推理：一步使用 (0,1)；多步使用连续区间，上一段输出作为下一段输入。
统一目标中心与长度尺度；目标、场景与关键点使用同一坐标变换，单位方向不做平移或缩放，旋转增强时同步旋转。输入均为真实表面采样点，当前环境包含桌面。编码器与方向几何偏置定义均与 FM 相同，但学生采用独立参数副本。
FlowMap.2 状态嵌入、8 层主干与映射头
K_a [B,J,3]
→ Linear3→128 → SiLU → Linear128→512
→ + KeypointIDEmbedding(J,512) → H_0 [B,J,512]
由 K_a、固定方向及 token 中心计算两路 (s,ρ)
→ ConditionalMapBlock ×8（层间参数不共享）
  1. AdaLN(e_map) → keypoint self-attention → gated residual
  2. AdaLN(e_map) → cross-attention(Q=H, K/V=G_obj, logits 加 b_obj) → gated residual
  3. AdaLN(e_map) → cross-attention(Q=H, K/V=G_scene, logits 加 b_scene) → gated residual
  4. AdaLN(e_map) → Linear512→2048 → GELU → Linear2048→512 → gated residual
→ LayerNorm512 → Linear512→3 → u_K [B,J,3]
→ K_b = K_a + (τ_b−τ_a) u_K
这里采用残差映射参数化。u 表示区间平均位移率，不是 FM 的瞬时速度 v；输出的最终语义由外层残差映射定义。τ_a=τ_b 时外层恒等返回输入；不除以时间差训练，避免极短区间的数值放大。
$$\Phi_\theta(x_a,\tau_a,\tau_b,C)=x_a+(\tau_b-\tau_a)u_\theta(x_a,\tau_a,\tau_b,C).$$
四个 modulation head 分别为 SiLU → Linear(512,1536)，各输出 γ、β、gate；AdaLN=(1+γ)⊙LN(H)+β。注意力 Q/K/V 和输出投影均为 512→512，8×64 heads；dropout=0。掌面间距与时间对改变每层 AdaLN 调制；方向通过各层两路 DirectionBiasMLP 注入。几何 tokens 与中心在同一场景采样中缓存；每次映射调用用该段输入 K_a 重算 (s,ρ)，同一次调用的各层复用关系张量，再由各层 MLP 分别计算偏置。
FlowMap.3 教师监督与推理
训练：冻结 FM 完整教师
noise K_0 + 固定条件 C（点云、掌面间距、方向） → 教师 ODE 轨迹
→ 抽取同一轨迹的 K_a^T、K_b^T
学生输入 K_a^T, τ_a, τ_b, C → K_b^S
→ mean squared error(K_b^S, stopgrad(K_b^T))

推理：K_0 ~ N(0,I)
一步：KeypointMap(K_0,0,1,C) → K_1
多步：按时间网格连续调用映射；不再用 FM Euler 更新
终点恢复米制坐标 → 掌部 SVD → 手指 IK → (R,t,q)
→ 掌面间距误差、方向角误差、限位、重建误差、网格碰撞与物理成功验收
FM 训练的独立噪声—真值线性插值只是构造速度监督；FlowMap 的映射监督必须来自同一教师 ODE 轨迹。不要把任意两点线性插值直接当成已训练教师的流映射。最终关键点仍需满足刚性手掌和可达手形要求，映射网络不会自动保证 IK 成功。
```
