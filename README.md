# SkillEvolve: Visual Prompt Evolution & Optimization Framework

SkillEvolve is an agentic framework for automatically optimizing visual prompts for LLM-based computer vision tasks, using iterative reflective training loops.

## Features
- 🎨 Automatic visual prompt optimization for classification, detection, and description tasks
- 🔄 Multi-model support (OpenAI GPT-4o, Claude 3, Qwen local models, etc.)
- 📊 Built-in evaluation and performance tracking
- ⚡ Configurable optimization loops with custom thresholds
- 🖥️ Optional WebUI dashboard for monitoring progress

## Installation
```bash
# Install core dependencies
pip install -e .

# Install with optional dependencies (e.g. Claude support, dev tools)
pip install -e ".[claude,dev]"
```

## Quick Start
1. Copy `.env.example` to `.env` and fill in your API keys
2. Run training:
```bash
skillevolve-train
```
3. Run evaluation:
```bash
skillevolve-eval
```

## Project Structure
```
SkillEvolve/
├── skillevolve/          # Core package
│   ├── optimizer.py      # Main prompt optimization logic
│   ├── engine/           # Training loop engine
│   ├── evaluation/       # Evaluation metrics
│   ├── gradient/         # Prompt gradient calculation
│   └── models/           # Model backends
├── configs/              # Configuration files
├── datas/                # Training/validation datasets
├── scripts/              # CLI entry points
├── tests/                # Unit tests
├── outputs/              # Optimization results
└── assets/               # Static assets
```

## Development
```bash
# Run linting
ruff check .

# Run tests
pytest tests/
```
