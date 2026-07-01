# SkillEvolve: LLM Agent技能自进化框架
SkillEvolve是一个面向大模型智能体的自动化技能演化优化框架，通过迭代式反思训练循环，自动优化Agent的领域技能Prompt与执行策略，实现技能能力的自主迭代升级。
## 核心特性
- 🧠 **自动化技能进化**：基于梯度反思、聚合优化机制，无需人工干预即可迭代提升Agent任务完成准确率
- 🔄 **全链路闭环训练**：支持Rollout采样→效果评估→梯度计算→策略更新→再验证的完整自进化闭环
- 🤖 **多模型适配**：兼容OpenAI/Anthropic/通义千问/豆包/本地开源模型（Qwen/DeepSeek/LLaMA等）
- 📊 **内置评估体系**：自带效果门控机制，自动过滤无效更新，保证技能进化方向稳定
- ⚙️ **高度可配置**：支持自定义训练轮次、优化阈值、数据集、评估指标
- 🛠️ **多模态能力扩展**：内置图像/视频/语音生成、搜索等工具接入能力，支持复杂多模态Agent技能进化
## 快速开始
### 环境依赖
- Python >= 3.12
- PyTorch >= 2.0
### 安装
```bash
# 安装核心依赖
pip install -r requirements.txt
# 或开发模式安装
pip install -e .
```
### 配置
1. 复制环境变量模板：`cp .env.example .env`
2. 在`.env`中填写你的模型API密钥、服务地址等配置信息
3. 配置训练参数：编辑`configs/default.yaml`自定义数据集、训练轮次、优化阈值等
### 运行训练
```bash
# 直接通过入口文件运行
python main.py
# 或通过脚本启动训练
python scripts/train.py
```
### 效果评估
```bash
python scripts/eval.py
```
## 项目结构
```
SkillEvolve/
├── engine/              # 核心训练引擎，实现训练循环、调度逻辑
│   └── trainer.py       # 技能训练主逻辑
├── optimizer/           # 技能优化模块，实现技能选择、更新、调度
│   ├── select.py        # 候选技能选择策略
│   ├── scheduler.py     # 优化调度器
│   └── skill.py         # 技能实体定义与处理
├── gradient/            # 技能梯度模块，实现反思、梯度聚合逻辑
│   ├── aggregate.py     # 多轮梯度聚合
│   └── reflect.py       # 失败案例反思逻辑
├── evaluation/          # 效果评估模块
│   └── gate.py          # 效果门控，判断是否接受技能更新
├── models/              # 模型适配层，统一不同大模型调用接口
│   ├── chat.py          # 对话模型接口
│   └── responses.py     # 响应模型接口
├── rollouts/            # 采样模块，生成任务交互轨迹
│   └── docvqa.py        # DocVQA任务采样实现
├── tools/               # 工具能力层，内置多模态工具封装
│   └── image.py         # 图像生成/处理工具
├── configs/             # 配置文件目录
│   └── default.yaml     # 默认训练配置
├── scripts/             # 命令行脚本
│   ├── train.py         # 训练启动脚本
│   └── eval.py          # 评估启动脚本
├── tests/               # 单元测试目录
├── datas/               # 数据集目录（.gitignore排除，需自行放置）
├── outputs/             # 训练输出目录（生成的技能、 checkpoint，.gitignore排除）
├── logs/                # 训练日志目录（.gitignore排除）
├── main.py              # 项目主入口
├── pyproject.toml        # 项目依赖与打包配置
├── requirements.txt     # 依赖列表
├── .env.example         # 环境变量模板
└── README.md            # 项目说明
```
## 分支规范
项目采用标准三分支架构：
- `main`：稳定生产分支，存放经测试验证的可运行版本
- `merge`：合并中转分支，用于dev分支合并到main前的冲突处理与测试
- `dev`：日常开发分支，所有新功能、bug修复先提交到dev分支
合并流程：`dev → merge → main`，合并冲突时优先保留来源分支内容。
## 开发规范
```bash
# 代码检查
ruff check .
# 运行单元测试
pytest tests/
```
## 注意事项
1. 所有敏感配置（API密钥、内部服务地址）请填写在本地`.env`文件中，禁止提交到Git仓库
2. 训练数据集、生成的技能 checkpoint、日志文件等大体积文件不会被提交到仓库，需本地自行管理
3. 本地开源模型部署推荐使用vLLM提供OpenAI兼容接口，可直接适配框架的模型调用层
