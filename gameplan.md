# Project Improvement Gameplan

## Goal
Transform this from a class assignment into a polished portfolio piece that demonstrates deep understanding of neural networks, professional Python practices, and rigorous validation. I am a Computer Engineering Masters student with a strong interest in AI/ML, and I want this project to reflect my capabilities and read like a research-level implementation.

---

## Phase 1: Structural Cleanup
> Fix naming, imports, and project packaging so the repo looks professional at first glance.

- [x] Rename `test.py` → `train.py`
- [x] Convert all bare imports to relative imports (`from .config import ...`)
- [x] Restructure the package to `src/custom_nn/` with `__init__.py`
- [x] Add `pyproject.toml` for pip-installable package (`pip install -e .`)
- [x] Update `requirements.txt` (unpin torch to reasonable range, keep torchvision for dataset loading)
- [x] Add `.gitignore` (data/, __pycache__/, *.pyc, .venv/, etc.)

---

## Phase 2: Refactor into a Clean API
> Make the neural network a reusable class rather than procedural scripts.

- [x] Create a `NeuralNetwork` class with methods:
  - `__init__(config)` — build layers from config
  - `forward(X, training=True)` — forward pass
  - `backward(y_true)` — backward pass
  - `train(X_train, Y_train, X_val, Y_val)` — full training loop
  - `predict(X)` — inference
  - `evaluate(X, Y)` — return loss + accuracy
  - `save(path)` / `load(path)` — serialize weights with `np.save`/`np.load`
- [x] Move training loop logic out of the script and into the class
- [x] Keep `train.py` as a thin entry point that instantiates and calls the class

---

## Phase 3: PyTorch Comparison Implementation
> Build an equivalent model in PyTorch and compare results side-by-side.

- [ ] Create `src/PyTorchNN/` directory with:
  - `model.py` — PyTorch model matching the custom architecture (same layers, sizes, activations)
  - `train.py` — training script using PyTorch's built-in optimizer, loss, etc.
  - `__init__.py`
- [ ] Use identical hyperparameters (learning rate, batch size, epochs, etc.)
- [ ] Use identical data splits (same seed, same preprocessing pipeline)
- [ ] Log training/validation loss per epoch for both implementations
- [ ] Generate comparison plots:
  - Training loss curves (both on same plot)
  - Validation loss curves (both on same plot)
  - Final test accuracy table

---

## Phase 4: Testing & Correctness Validation
> Prove the math is correct with automated tests.

- [ ] Create `tests/` directory with pytest structure
- [ ] Add gradient checking tests (finite difference vs. analytic gradients) for:
  - Dense layer (weights, biases)
  - ReLU backward
  - Softmax + Cross-entropy backward
  - Batch Normalization backward (gamma, beta, inputs)
- [ ] Add unit tests for:
  - Forward pass output shapes
  - Dropout mask behavior (training vs. inference)
  - Early stopping trigger logic
  - ADAM optimizer parameter updates
- [ ] Add integration test: train for 2 epochs, verify loss decreases
- [ ] Add PyTorch vs. Custom comparison test (same input → similar output within tolerance)

---

## Phase 5: CLI & Experiment Support
> Allow running experiments from the command line with different configurations.

- [ ] Add `argparse` CLI to `train.py`:
  - `--epochs`, `--lr`, `--batch-size`, `--hidden-layers`, `--hidden-units`
  - `--seed`, `--save-path`, `--no-early-stopping`
- [ ] Save training metrics to CSV/JSON after each run
- [ ] Add model checkpoint saving (best validation loss weights)

---

## Phase 6: Visualization & Analysis
> Go beyond a single loss curve — show the model's behavior in detail.

- [ ] Confusion matrix heatmap on test set
- [ ] Per-class accuracy bar chart (which clothing items are hardest?)
- [ ] Sample predictions grid (correct and incorrect, showing images)
- [ ] Training curve comparison plot (custom vs. PyTorch)
- [ ] Save all figures to `results/` with descriptive names

---

## Phase 7: Documentation & README
> Rewrite the README to tell a story and showcase the work.

- [ ] Project overview with motivation (why build from scratch?)
- [ ] Architecture diagram (layer sizes, activations, techniques)
- [ ] Math section: key equations for forward/backward pass (LaTeX in markdown)
- [ ] Hyperparameter table with final chosen values and brief justification
- [ ] Results section:
  - Custom NN vs. PyTorch accuracy comparison table
  - Training curves (both implementations)
  - Confusion matrix
  - Per-class breakdown
- [ ] Discussion: what matched, what differed, lessons learned
- [ ] Installation & usage instructions (updated for new CLI)
- [ ] Add type hints and docstrings to all public classes/methods

---

## Summary & Priority

| Phase | Impact | Effort | Status |
|-------|--------|--------|--------|
| 1. Structural Cleanup | Medium | Low | Completed |
| 2. Refactor to Class API | High | Medium | Completed |
| 3. PyTorch Comparison | Very High | Medium | Not Started |
| 4. Testing & Validation | High | Medium | Not Started |
| 5. CLI & Experiments | Medium | Low | Not Started |
| 6. Visualization | High | Low | Not Started |
| 7. Documentation & README | Very High | Medium | Not Started |

---

## Notes
- Phases 1-2 should be done first (they unblock everything else)
- Phases 1-2 are now complete: the codebase is packaged under `src/custom_nn/`, exposes `NetworkConfig` + `NeuralNetwork`, and supports both `python train.py` and `python -m custom_nn`
- Phase 3 is the marquee feature — most impressive to reviewers
- For Phase 3, reuse the exact same preprocessed train/validation/test split for both implementations instead of preprocessing separately, otherwise the comparison will not be truly apples-to-apples
- `results/` has replaced `images/` as the output directory; future plots, checkpoints, and comparison tables should land there
- `NetworkConfig` currently coexists with legacy module-level constants in `config.py`; Phase 3 code should prefer the config object to avoid reintroducing global wiring
- Phase 4 is what separates this from "followed a tutorial"
- Phases can overlap (e.g., write docs as you build each feature)
