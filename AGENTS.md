# AGENTS.md

## Project Overview

Custom Neural Network implementation from scratch (no ML frameworks) for Fashion MNIST classification. Uses NumPy for computation, PyTorch only for data loading.

See [gameplan.md](gameplan.md) for the planned improvement roadmap (7 phases).

## Architecture

```
src/CustomNN/
├── config.py              # All hyperparameters (single source of truth)
├── data_preprocessing.py  # Fashion MNIST loading & preprocessing pipeline
├── model.py               # Layer_Dense, Activation_ReLU, Activation_Softmax, loss
├── techniques.py          # ADAM_Optimizer, Early_Stopping, Dropout, Batch_Normalization
└── __init__.py            # Package exports with relative imports
test.py                    # Main training script (entry point)
```

**Data flow**: Load Fashion MNIST → NumPy arrays → standardize (train stats only) → mini-batch training → forward/backward/ADAM update → early stopping

## Conventions

- **Classes**: `PascalCase_With_Underscores` (e.g., `Layer_Dense`, `Activation_ReLU`)
- **Methods**: All layers implement `forward()` and `backward()` — preserve this pattern
- **Config constants**: `UPPER_SNAKE_CASE` in `config.py` — never hardcode hyperparameters elsewhere
- **Layer caching**: Layers store `self.inputs`, `self.output`, `self.dinputs` for backprop
- **Inverted dropout**: Scale by `1/keep_prob` during training, identity at inference

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run training (downloads ~200MB Fashion MNIST on first run)
python test.py
```

## Known Issues

- `test.py` uses bare imports (`from model import...`) — conflicts with package-style relative imports in `__init__.py`. When running as script, execute from the `src/CustomNN/` directory or adjust imports.
- Training functions use global state (`layers` list, optimizer objects) — not class-based yet.
- `plt.show()` at end of training blocks until figure window is closed.
- No `pyproject.toml` or installable package setup yet.

## Key Design Decisions

- **No ML library usage**: All forward/backward math is hand-implemented in NumPy. Do NOT replace with PyTorch/TensorFlow equivalents.
- **Centralized config**: All hyperparameters live in `config.py`. New parameters must be added there.
- **Standardization**: Computed from training set only, applied to val/test (prevents data leakage).
- **Batch norm momentum**: Set to 0.185 (intentionally low vs typical 0.9–0.99) — do not "fix" without discussion.

## When Adding New Layers or Techniques

1. Implement `forward(self, inputs, training=True)` and `backward(self, dvalues)`
2. Cache necessary values in `self.*` during forward for backward pass
3. Export from `__init__.py`
4. Add any new hyperparameters to `config.py`
5. Integrate into the `layers` list in `test.py`
