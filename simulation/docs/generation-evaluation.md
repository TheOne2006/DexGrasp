# Sharpa 预测评估与数据生成

`data / train / test` 保持原有职责。新增 `simulation/` 是本仓库维护、独立安装的
Dexonomy 仿真运行时；不依赖外部 reference checkout，不改变训练网络。
训练环境的 NumPy <2 与仿真固定 NumPy 2.2.6 分开，通过 JSON 传递预测。

## 安装和外部资产

按 [simulation README](../README.md) 建立 Python 3.11 独立环境。
固定 MuJoCo/MJWarp 3.17.0、Warp 1.16.0；Torch 按硬件环境安装。
4090 实测锁与 5090 待验证方案见 [环境说明](../environments/README.md)。
CPU 诊断不用 CUDA；GPU 必须通过下述带三锁、PID 归属和资源余量检查的 runner。
训练包的普通 import、评估 `--help`、输入检查均不初始化 Warp。

自行提供具有使用权限的 Sharpa `right_kp.xml` 及其引用网格；生产还需要完整
35 抓型模板、骨架和关键点标注。`dexonomy.cli.check_assets` 按固定清单检查资产。
场景包含物体 XML、表面/碰撞几何、info、pose 和 scale。不得用视觉网格替代
碰撞资产。未随代码发布模型、模板、数据、权重。来源及许可见
[NOTICE](../NOTICE.md) 和 [提取清单](../SOURCES.json)。

## 预测 → 明确的物理输入

先使用匹配 checkpoint 的源码得到 K21，按该 checkpoint 的手标定解码 R/t/q。
`dexgrasp.test.physics_input.export_request` 接收 decoded JSONL，以及调用者明确提供的：

- `protocol`: `floating` 或 `tabletop`。
- `joint_names`: 22 个 q 关节名；运行时按实际 MJCF 编译顺序映射，不按数组长度猜。
- `scenes`: scene_id 到可信本地场景文件的映射。
- `controls`: 每个 `(sample_index, candidate_index)` 对应 `pregrasp`、`squeeze`。
  两者可以是一个 `{R,t,q}`，也可以是多个姿态组成的有序列表。
  控制必须已位于世界坐标、官方 hand root；保留完整轨迹，不自动生成或截断 squeeze。
- `control_source`: 控制策略或固定来源的版本标识。
- `frame_transforms`: 每个 scene_id 的 `world_from_input_R`、`world_from_input_t`、
  `official_from_decoded_R`，均显式给出；已经是世界/官方手坐标时给单位变换。

转换为 `R_world_official = R_world_input @ R_decoded @ A.T`、
`t_world = R_world_input @ t_decoded + t_world_input`。关节角不变。
对象局部输入已经以米表示，**不能再次乘 object scale**；scale 只由物体场景加载。
四元数在运行时由旋转矩阵转换为 MuJoCo 的 wxyz，JSON 不接受隐含 xyzw。
失败 IK 保留为 invalid_input，不丢出分母。grasp 角度需在模型限位内；
squeeze 是位置控制目标，允许沿用明确策略的越限目标，模拟中的关节约束仍保留。

运行时请求 schema 为 `dexgrasp-physics/v1`，声明 `frame=world`、
`hand_pose_frame=official_hand_root`、`units={length:m,angle:rad}`，并绑定
`decoded_sha256`。每个候选携带 sample/candidate 索引、scene 以及三段 R/t/q。
相对资产路径以请求文件所在目录解析。原生 Dexonomy `.npy` 场景使用 pickle，
只对可信本地资产显式传入 `--trusted-scenes`，不加载外来未验证 pickle。

```bash
.venv-simulation/bin/python -m dexonomy.prediction_eval request.json \
  --check-only --output input-check.json
.venv-simulation/bin/python -m dexonomy.prediction_eval request.json \
  --trusted-scenes --backend cpu --output physics.json
python -m dexgrasp test-physics --decoded decoded.jsonl --report physics.json
```

CPU 与 Warp 都输出逐候选 success/failure/invalid_input/simulation_error、逐方向指标、
输入与场景/模型/源码身份、依赖版本和时长。存在无效或异常候选时完整成功率为 null，
同时报告覆盖率和已验证成功数。碰撞未独立检测时 `collision_free=null`。
`test-physics` 不把旧射线间距合格率拼成物理联合成功率；原 `test --verdicts` 接口保留。

## 两个固定物理协议

Floating：场景只有自由物体，不得借桌面支撑。每一方向从独立 reset 重放完整
pregrasp/grasp/squeeze，再执行 ±X/±Y/±Z 六向外力。每方向 10 次、每次 50 步；
质量 0.1 kg、力为质量 ×9.8、摩擦 0.6/0.02、noslip0，位移 <5 cm、旋转 <15°。
六方向都报告，即使前一个方向已失败。

Tabletop：保留原桌面，先执行官方 +Z 10 cm Slide。只有 Slide 通过的候选继续；
每个 FC 方向从同一完整 Slide 终态恢复（包括速度、控制、warmstart 等），不重置为
grasp_qpos。六方向互不继承状态；保留桌面再接触失败判据和逐方向快照哈希。
未通过 Slide 的候选没有 FC 数据，不能伪填六个失败测量。

## h32 / K21 与当前训练代码的边界

已交单物体数据：1024 个目标点、交接 K21、q22，h 是官方掌面 32 点到真实物体
表面的平均最近距离（原始米制），方向是官方手 +Zh。对象局部坐标和手标定旋转 A
需按上面的变换还原。K21 必须使用交接 hand_spec，不能只凭点数相同沿用旧标定。
若预测仍为归一化 K，先用训练时的中心 o 和尺度 L 恢复 `K_m=K_normalized*L+o`。

远端集成验证时的 main 使用场景点云、射线 h，且当时三份模型/推理/解码核心源码与所提供
checkpoint provenance 的哈希不匹配。因此本次没有改网络来冒充匹配，也没有报告
checkpoint 推理或其物理成功率。这里的物理入口接受独立解码结果；匹配源码到位后
可直接接入。合并后的训练接口以仓库根目录 README 和 docs/data.md 为准，历史验证不代表当前 checkpoint 已通过验收。h32 不被重解释为射线命中中位数，+Zh 不被默认为解码坐标系第三列。

## 本机有界生成

从 `simulation/examples/unified-night/` 的对应协议模板填写本机配置。外部 index
每个所选物体/pose 必须保留六个 scale，不能静默丢掉不易处理的输入。
配置必须包含本 checkout `simulation/` 的绝对 source、独立 python、asset_root、
独立 output/checkpoint/cache、index 哈希、单 GPU UUID、三把已有兼容锁、有限
case_limit_s（可选正数）、未来 hard_deadline_epoch、stop_new_init_epoch。
默认设置 `world_size=1, rank=0, execution_mode=sequential, continuous=false`。
产品也支持显式 `continuous=true`，此时 deadlines/tours/max_group_visits/case_limit_s
均须为 null。长期运行是调用方的独立决定，不继承本次验证授权。本任务实验仍遵守
每次 ≤600 秒、累计 ≤3600 秒；这个实验预算不写死在通用入口。
示例的 `deployment_template_only=true` 必须在资源分配后显式改为 false。

```bash
.venv-simulation/bin/python -m dexonomy.generation check local-config.json
.venv-simulation/bin/python -m dexonomy.generation run local-config.json --name attempt-01
.venv-simulation/bin/python -m dexonomy.generation read local-config.json
.venv-simulation/bin/python -m dexonomy.generation resume local-config.json --name attempt-02
```

入口同步运行已有 runner：三锁、两次 CUDA context PID challenge、未知 PID 拒绝、
GPU/主机/cgroup/shm 余量 guard 均保留。PID challenge 之后先执行环境预检，
检查 Torch/CUDA 与实际 GPU 架构匹配；记录设备总显存、驱动和运行时版本。只停止本次拥有的进程组。
同一输入、源码和配置身份才可恢复；恢复不是另一次从零生产。不要改源码后继续旧账本。
严格 Init、有限补采、顺序 Refine/Eval、成功模板反馈、stage 事务与持久化均复用固定来源。
退出 0 不等于物理成功；读取 `summary.json`、`state.npy`、各 job 的 Eval/feedback
及产物哈希后验收。原生 NPY 结果仅用于可信本地任务；不上传 Git。

GPU 预测也使用 `simulation/scripts/grouped_production/run.py`：配置 worker 为
`prediction_worker.py`、prediction_request 为请求路径，外加上述资源/超时/三锁字段；
省略生产 checkpoint_schema。结果保存在该 attempt 的 `data/physics.json`。
直接 `--backend warp` 是底层 API，调用者仍须先通过相同的外层资源 guard。

验证范围与实测结果见 [bounded validation](generation-evaluation-validation.md)。
