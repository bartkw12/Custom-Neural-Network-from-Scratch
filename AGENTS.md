# AGENTS.md

## Project Overview

Custom Neural Network implementation from scratch (no ML frameworks) for Fashion MNIST classification. Uses NumPy for computation, PyTorch only for data loading.

See [gameplan.md](gameplan.md) for the planned improvement roadmap (7 phases).

## Architecture

```
src/custom_nn/
├── cli.py                 # Unified CLI (custom, pytorch, compare, analyze)
├── config.py              # Hyperparameters + NetworkConfig dataclass
├── data_preprocessing.py  # Fashion MNIST loading & preprocessing pipeline
├── model.py               # Layer_Dense, Activation_ReLU, Activation_Softmax, loss
├── network.py             # NeuralNetwork class encapsulating training/eval/save/load
├── techniques.py          # ADAM_Optimizer, Early_Stopping, Dropout, Batch_Normalization
├── __init__.py            # Public package exports
└── __main__.py            # Package entry point for `python -m custom_nn`
src/pytorch_nn/
├── model.py               # PyTorch reference network matching custom architecture
├── train.py               # PyTorch training/evaluation helpers
├── compare.py             # Comparison artifact helpers (loss/error plots)
├── analyze.py             # Analysis visualizations (confusion matrix, per-class, samples)
└── __init__.py
train.py                   # Thin training script (repo entry point)
run_pytorch.py             # Compatibility entry point for PyTorch runs
results/                   # Latest/archive run summaries, checkpoints, and figures
```

**Data flow**: Load Fashion MNIST → NumPy arrays → standardize (train stats only) → `NeuralNetwork.train()` mini-batch loop → forward/backward/ADAM update → early stopping → persist summaries/checkpoints → compare/analyze artifact generation

## Conventions

- **Classes**: `PascalCase_With_Underscores` (e.g., `Layer_Dense`, `Activation_ReLU`)
- **Methods**: All layers implement `forward()` and `backward()` — preserve this pattern
- **Config constants**: `UPPER_SNAKE_CASE` in `config.py` remain available; `NetworkConfig` is the structured API for wiring the model
- **Layer caching**: Layers store `self.inputs`, `self.output`, `self.dinputs` for backprop
- **Inverted dropout**: Scale by `1/keep_prob` during training, identity at inference

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run training (downloads ~200MB Fashion MNIST on first run)
python train.py

# Or run the package entry point
python -m custom_nn

# Explicit workflows
python train.py custom
python train.py pytorch
python train.py compare
python train.py analyze
```

## Known Issues

- `train.py` prepends `src/` to `sys.path` so it can run directly from the repo root before installation. After `pip install -e .`, package imports work without that fallback.
- `NetworkConfig` currently coexists with legacy module-level constants in `config.py` as a transition step.

## Key Design Decisions

- **No ML library usage**: All forward/backward math is hand-implemented in NumPy. Do NOT replace with PyTorch/TensorFlow equivalents.
- **Centralized config**: Hyperparameter defaults live in `config.py`. New parameters must be added there and surfaced through `NetworkConfig` when they affect model wiring or training.
- **Standardization**: Computed from training set only, applied to val/test (prevents data leakage).
- **Batch norm momentum**: Set to 0.185 (intentionally low vs typical 0.9–0.99) — do not "fix" without discussion.

## When Adding New Layers or Techniques

1. Implement `forward(self, inputs, training=True)` and `backward(self, dvalues)`
2. Cache necessary values in `self.*` during forward for backward pass
3. Export from `__init__.py`
4. Add any new hyperparameters to `config.py` and thread them through `NetworkConfig` when needed
5. Integrate into `NeuralNetwork._build_layers()` in `network.py`
