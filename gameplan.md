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

- [x] Create `src/pytorch_nn/` package with:
  - `model.py` — PyTorch model matching the custom architecture (same layers, sizes, activations)
  - `train.py` — training helpers using PyTorch's built-in optimizer, loss, scheduler, and evaluation flow
  - `compare.py` — history loading helpers for the comparison workflow
  - `__init__.py`
- [x] Use identical hyperparameters (learning rate, batch size, epochs, etc.)
- [x] Use identical data splits (same seed, same preprocessing pipeline)
- [x] Log training/validation loss per epoch for both implementations
- [ ] Generate comparison plots:
  - Training loss curves (both on same plot)
  - Validation loss curves (both on same plot)
  - Final test accuracy table

Final generation notes in Phase 3 Implementation chat. 

---

## Phase 4: Testing & Correctness Validation
> Prove the math is correct with automated tests.

- [x] Create `tests/` directory with pytest structure
- [x] Add gradient checking tests (finite difference vs. analytic gradients) for:
  - Dense layer (weights, biases)
  - ReLU backward
  - Softmax + Cross-entropy backward
  - Batch Normalization backward (gamma, beta, inputs)
- [x] Add unit tests for:
  - Forward pass output shapes
  - Dropout mask behavior (training vs. inference)
  - Early stopping trigger logic
  - ADAM optimizer parameter updates
- [x] Add integration test: train for 2 epochs, verify loss decreases
- [x] Add PyTorch vs. Custom comparison test (same input → similar output within tolerance)

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
| 3. PyTorch Comparison | Very High | Medium | In Progress |
| 4. Testing & Validation | High | Medium | Completed |
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
- Phase 3 is now partially implemented: `src/pytorch_nn/` exists and exposes a PyTorch model, shared-data training helpers, history saving, test evaluation, and comparison-history loading
- The PyTorch implementation currently mirrors the custom config surface: same architecture shape, same seed, same preprocessing pipeline, same batch size, same learning rate, same ADAM betas/epsilon/weight decay, and per-batch learning-rate decay
- The PyTorch training path auto-detects `cuda`, then `mps`, then `cpu`, so dedicated GPU training is supported when PyTorch is installed with CUDA support
- The custom training entry point now writes `results/custom_nn_history.json`, and the PyTorch helpers write `results/pytorch_history.json`, so the data needed for side-by-side plotting is available
- Remaining Phase 3 work is the actual visualization/reporting layer: generate overlaid loss plots, produce a final accuracy comparison artifact, and add a cleaner top-level PyTorch run entry point if desired
- Phase 4 is what separates this from "followed a tutorial"
- Phases can overlap (e.g., write docs as you build each feature)
