# DexGrasp 自有代码仓库

- 本目录是 Git/GitHub 根目录。上层是研究工作区，`../reference/` 中每个项目是独立上游仓库，不是本仓库子模块。
- 三个业务部分统一称为 data、train、test。自有最终评估划分称 `test`；数据适配层保留上游 `test.json`，兼容早期 `evaluation` 标签。`tests/` 仅用于软件单元测试。
- 项目结构见 `docs/project-structure.md`，参考代码用法见 `docs/reference-code.md`；参考清单记录来源和接口，不要求固定上游 commit。
- 自有模型训练不依赖 reference；FK/IK 等显式适配器按需使用参考代码。移植代码时保留来源说明及对应版权/许可证文本，业务修改在本仓库落地。
- 数据、权重、结果使用上层 `data/`、`checkpoints/`、`outputs/`。不要将其提交到 GitHub。
- 不修改其他 EPIC 项目、共享环境或系统设置。没有明确授权时不提交、不推送、不启动正式训练。
- 修改数据或评估接口后运行相关软件测试；不得把 IK 拟合成功当作物理抓取成功。当前训练循环单元测试禁用 optimizer.step。
