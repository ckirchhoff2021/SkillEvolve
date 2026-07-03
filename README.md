# SkillEvolve: LLM Agent技能自进化框架
SkillEvolve是基于**ReflACT（Reflective Action 反思行动）** 迭代训练流水线的智能体技能自动化优化框架，无需微调大模型权重，即可通过闭环迭代自动优化文档问答、图像分类、目标检测等任务的结构化技能提示文档，实现Agent任务能力的自主迭代升级。
## 核心特性
- 🧠 **全自动6阶段进化流水线**：内置完整ReflACT闭环：Rollout轨迹采样 → 成败案例反思 → 优化补丁聚合 → 高价值补丁选择 → 技能更新 → 效果门控验证
- 🎨 **视觉任务原生支持**：开箱支持DocVQA文档视觉问答任务，可快速扩展至图像分类、目标检测、多模态描述等各类CV任务
- 🔄 **全模型兼容**：适配所有OpenAI接口格式兼容的大模型，包括OpenAI GPT-4o、Anthropic Claude、通义千问、豆包，以及本地部署的Qwen/DeepSeek/LLaMA等开源模型
- ✅ **零微调成本**：仅优化技能提示文档，无需修改或微调任何模型权重，训练成本极低、迭代速度快
- 📊 **内置效果保障机制**：自动在验证集上评估优化效果，仅保留准确率提升的技能更新；支持自动checkpoint保存、训练早停、效果指标追踪
- ⚙️ **高度可扩展**：模块化设计，支持自定义任务类型、评估指标、优化策略、阈值参数
## 环境依赖
- Python >= 3.12
- 基础依赖见`requirements.txt`/`pyproject.toml`
## 快速开始
### 1. 安装依赖
```bash
# 安装核心依赖
pip install -r requirements.txt
# 开发模式安装（支持代码修改即时生效）
pip install -e .
```
### 2. 配置环境变量
```bash
# 复制环境变量模板
cp .env.example .env
# 编辑.env文件，填写你的模型API地址、密钥、模型名称等配置
```
### 3. 准备数据与配置
1. 在`datas/docvqa/`目录下放置：
   - DocVQA数据集文件
   - 初始技能文件`init_skill.md`（基础提示词）
   - 训练/验证样本索引文件`sample.json`
2. 根据需求编辑`configs/default.yaml`，配置训练轮次、优化阈值、样本数量等参数
### 4. 运行训练与评估
```bash
# 启动技能进化训练
python main.py
# 或通过脚本启动
python scripts/train.py
# 评估当前技能效果
python scripts/eval.py
```
## 核心进化流程
SkillEvolve通过多轮迭代闭环实现技能自动优化，每轮执行以下步骤：
1. **Rollout采样**：使用当前技能在训练集上运行目标模型，收集成功/失败交互轨迹
2. **Reflect反思**：分析失败案例原因、总结成功案例经验，生成结构化技能修改补丁
3. **Aggregate聚合**：合并多轮反思生成的补丁，消除冲突、保留共性有效优化点
4. **Select选择**：对补丁进行价值排序，在token预算内选择高影响的优化点
5. **Update更新**：应用选中的补丁，生成新版本候选技能文档
6. **Evaluate门控**：在验证集上测试候选技能效果，仅当效果优于当前版本时接受更新，否则丢弃
流程自动重复直到达到目标准确率阈值或触发早停。
## 项目结构
```
SkillEvolve/
├── configs/              # 配置文件目录
│   └── default.yaml      # 默认训练配置（轮次、阈值、数据集路径等）
├── engine/               # 核心训练引擎，实现ReflACT流水线主逻辑
│   └── trainer.py        # 训练调度主入口
├── models/               # 模型适配层，统一OpenAI对话/响应接口，兼容多模型调用
├── rollouts/             # 任务采样模块，原生支持DocVQA任务，可扩展其他任务
│   └── docvqa.py         # DocVQA任务采样实现
├── gradient/             # 梯度模块，实现成功/失败案例反思、补丁聚合逻辑
├── optimizer/            # 技能优化模块，包含补丁选择、技能更新、学习率调度器
├── evaluation/           # 效果评估与门控模块，判断是否接受新技能更新
├── tools/                # 工具函数库，包含日志、图像处理、通用工具等
├── scripts/              # 命令行入口脚本
│   ├── train.py          # 训练启动脚本
│   └── eval.py           # 评估启动脚本
├── tests/                # 单元测试目录
├── datas/                # 数据集目录（.gitignore排除，需自行放置数据、初始技能、索引）
├── prompts/              # 进化流程Prompt模板，包含反思/聚合/排序/重写等各阶段提示词
├── outputs/              # 训练输出目录，自动保存生成技能、checkpoint（.gitignore排除）
├── assets/               # 静态资源目录
├── main.py               # 项目主入口文件
├── .env.example          # 环境变量模板（所有值已清空，填写后使用）
├── .gitignore            # Git忽略规则
├── pyproject.toml        # 项目打包与依赖配置
├── requirements.txt      # 依赖列表
└── README.md             # 项目说明文档
```
## 分支规范
项目采用标准三分支架构管理版本：
- `main`：稳定生产分支，存放经测试验证的可运行版本
- `merge`：合并中转分支，用于dev分支合并到main前的冲突处理与测试
- `dev`：日常开发分支，所有新功能、bug修复先提交到dev分支
合并流程：`dev → merge → main`，合并冲突时优先保留来源分支内容。
## 开发规范
```bash
# 代码格式检查
ruff check .
# 运行单元测试
pytest tests/
```
## 注意事项
1. 当前版本开箱仅支持DocVQA任务，新增任务类型需在`rollouts/`目录下实现对应Rollout类
2. 训练效果高度依赖初始技能质量和优化器模型能力，建议使用能力较强的大模型作为优化器
3. 所有敏感配置（API密钥、内部服务地址）请填写在本地`.env`文件中，禁止提交到Git仓库
4. 数据集、训练输出、日志、缓存等大体积文件默认被`.gitignore`排除，不会提交到远程仓库
5. 本地开源模型推荐使用vLLM部署，提供OpenAI兼容接口即可直接接入框架
## License
MIT License
