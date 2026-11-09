# 项目结构与 GitHub 边界

项目是两层目录，不是两层 Git 仓库。外层 `DexGrasp/` 是研究工作区；内层同名目录是自有代码仓库。保留现有路径，避免破坏已有配置和资料链接。

```text
<workspace>/DexGrasp/              外层：研究工作区，无 .git
├── AGENTS.md, README.md, docs/    项目背景、阅读资料与报告
├── reference/                    独立上游源码，只按需要使用
│   ├── README.md                 导航
│   ├── KPGrasp/.git/
│   ├── DexLearn/.git/
│   ├── DexGraspNet2/.git/
│   ├── Dexonomy/.git/
│   └── DexGraspBench/.git/
├── data/                         原始/处理后数据和 manifest
├── checkpoints/                  权重
├── outputs/                      运行日志、模型 test 结果、检查记录
├── gpu-watch/                    已有监控工具，独立于模型代码
└── DexGrasp/                     内层：Git/GitHub 根目录
    ├── .git/
    ├── .github/                  CI 与 PR 模板
    ├── AGENTS.md, README.md
    ├── configs/                  自有实验配置、参考代码目录
    ├── dexgrasp_lab/
    │   ├── data.py, prepare.py   data
    │   ├── models.py, methods.py, train.py  train
    │   ├── test.py              模型 test：采样、指标、IK 与物理测试
    │   └── config.py, references.py, __main__.py
    ├── scripts/                  参考源码准备工具
    ├── docs/                     跟随代码发布的接口与结构说明
    └── tests/                    软件单元测试，不是模型 test 数据
```

## 哪些内容进入 GitHub

内层的代码、配置、文档、测试和 `.github/` 进入自有仓库。外层的参考仓库、数据、权重、运行结果和阅读资料不随内层上传；参考源码不作为内层 Git submodule。上游自己的 `third_party/` submodule 保留其原有管理方式。

Git 操作必须在内层执行：`git -C /absolute/workspace/DexGrasp/DexGrasp ...`。外层不是仓库，不应在那里执行 `git init` 来包住所有数据和参考仓库。内层原有 `.gitignore`、README 等仍未提交；本次也没有创建提交或推送。

## GitHub 基础设置

- 已核实远端：[TheOne2006/DexGrasp](https://github.com/TheOne2006/DexGrasp)，当前可见性为 **public（公开）**，默认分支 `main`。这是用户已创建的仓库，保留其现有可见性设置。
- 内层 `origin` 已关联 `https://github.com/TheOne2006/DexGrasp.git`；外层仍是无 `.git` 的研究工作区。
- 内层本地提交身份沿用用户现有 Git 身份，并写入本仓库配置；未修改 global config。
- 已添加 `.gitattributes`、`.editorconfig`、PR 模板和 CPU CI。CI 只跑软件检查，不启动正式训练。
- 2026-11-08 已通过 GitHub 连接器核实仓库归属和访问权限。本次仅关联远端、更新状态，没有创建提交或推送代码；CI 尚未在 GitHub 运行。

本机尚无 GitHub CLI/API 登录，GitHub 连接器访问权限不等同于本机 Git 推送凭据。公开仓库的只读访问与后续推送认证应分别核验。

## 新机器或 GitHub clone 后

先创建外层工作区，再把自有仓库 clone 到内层。配置中的 `../../data` 等路径以配置文件位置为基准；`reference/` 相对内层是 `../reference/`，不依赖当前 shell 目录。

```bash
# 在已经 clone 的内层仓库运行；默认只检查，不下载、不拉取更新。
python scripts/setup_references.py
# 按需准备 FK/IK 的参考代码；不会自动初始化整个第三方依赖树。
python scripts/setup_references.py --clone KPGrasp
```

完整五仓库来源和接口见 [参考代码使用说明](reference-code.md)。原生监督训练不需要克隆五个参考仓库。代码目录可以单独管理，但 FK/IK、官方复现入口需要相应的外层 reference 和环境。

## 术语与职责

最终采用 `data / train / test`。数据划分为 `train / val / test`，`val` 用于选模型，`test` 用于冻结后的最终评估。CLI 使用 `python -m dexgrasp_lab test --samples ...`；原 `metrics` 命令作为兼容别名保留。输入中的早期 `evaluation` 划分标签归一化为 `test`，输出始终使用 `test`。

`tests/` 和 `test_*` 是 unittest 的软件测试发现约定，与模型 test 阶段分开。上游文件名、命令和评估协议保持原样，不为术语调整改写 reference 源码。
