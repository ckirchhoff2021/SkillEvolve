# SkillEvolve: Visual Prompt Evolution & Optimization Framework

SkillEvolve is an agentic framework for automatically optimizing visual skill documents for LLM-based computer vision tasks, using the ReflACT (Reflective Action) iterative training pipeline. It can evolve structured skill prompts for visual tasks like document VQA, image classification, and object detection without fine-tuning model weights.

## Features
- 🎨 Automatic visual prompt optimization for classification, detection, and description tasks
- 🧠 Implements ReflACT 6-stage pipeline: Rollout → Reflect → Aggregate → Select → Update → Evaluate
- 🔄 Multi-model support (OpenAI GPT-4o, Claude 3, Qwen local models, etc.)
- ✅ No model fine-tuning required: optimizes only skill prompt documents
- 📊 Built-in evaluation and performance tracking
- ⚡ Configurable optimization loops with custom thresholds
- 🖥️ Optional WebUI dashboard for monitoring progress
- 💾 Automatic checkpoint saving and early stopping
- 🔧 Modular design, easy to extend for new visual tasks

## Installation
```bash
# Install core dependencies
pip install -e .

# Install with optional dependencies (e.g. Claude support, dev tools)
pip install -e ".[claude,dev]"

# Install all optional dependencies
pip install -e ".[all]"
```

## Quick Start
1. Copy `.env.example` to `.env` and fill in your API keys
```bash
cp .env.example .env
# Edit .env to fill API_URL, API_KEY, MODEL_NAME
```

2. Prepare data directory:
- Create `data/docvqa/` directory
- Place your DocVQA dataset under `data/DocVQA/`
- Add initial skill file `data/docvqa/init_skill.md` with your base visual skill prompt
- Add sample index file `data/docvqa/sample.json` with train/validate indices

3. Run training:
```bash
skillevolve-train
```
4. Run evaluation:
```bash
skillevolve-eval
```

## Core Concepts
SkillEvolve uses a fully automated reflective training loop to optimize visual skill prompts:
1. **Rollout**: Run target model on training set using current skill
2. **Reflect**: Analyze failures and successes to generate structured edits (patches)
3. **Aggregate**: Merge multiple patches into a consistent set of changes
4. **Select**: Rank and select high-impact edits within budget
5. **Update**: Apply edits to generate a new candidate skill
6. **Evaluate**: Run on validation set to decide whether to accept/reject the new skill

The process repeats until target performance threshold is reached or early stopping triggers.

## Project Structure
```
SkillEvolve/
├── configs/              # Configuration files
├── engine/               # Training engine, core ReflACT pipeline implementation
├── models/               # Model wrappers for OpenAI Chat/Responses API
├── rollouts/             # Task-specific rollout implementations (default: DocVQA)
├── gradient/             # Reflect and aggregate modules for generating skill edits
├── optimizer/            # Patch selection, skill update, and learning rate scheduler
├── evaluation/           # Validation gate and scoring functions
├── tools/                # Utility functions (image processing, logging, etc.)
├── scripts/              # Training and evaluation CLI entry points
├── tests/                # Unit tests
├── data/                 # Training data, initial skills, dataset indices (create this directory first)
├── prompts/              # LLM prompt templates for reflection/merge/rewrite (create this directory first)
├── outputs/              # Optimization results (generated automatically)
├── .env.example          # Environment variable template
├── pyproject.toml        # Project dependency configuration
└── README.md             # This file
```

## Development
```bash
# Run linting
ruff check .

# Run tests
pytest tests/
```

## Known Limitations & Notes
1. Currently only supports DocVQA task out of the box, can be extended by adding new Rollout implementations under `rollouts/`
2. First run requires creating `data/` and `prompts/` directories with required files
3. Default configuration assumes DocVQA dataset format, modify `rollouts/docvqa.py` for custom datasets
4. Training performance heavily depends on the quality of the initial skill prompt and the optimizer model capability

## License
MIT License
