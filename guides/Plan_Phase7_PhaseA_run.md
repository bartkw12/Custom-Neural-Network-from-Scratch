# Phase 7 - Phase A Run Plan (Generate Real Artifacts)

This runbook is for your local PC (Windows + GPU) to generate all artifacts needed for the README results section.

## 0) Open PowerShell in repo root

```powershell
cd "C:\Users\gts\Desktop\Platforms Sandbox\Use Case No37\Use Case No37 Testing\test\DL\Custom-Neural-Network-from-Scratch"
```

## 1) Prepare environment

Use your existing virtual environment if available.

```powershell
# If .venv already exists
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

If you do not have `.venv` yet:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

## 2) Verify PyTorch sees CUDA

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda devices:', torch.cuda.device_count()); print('device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Expected: `cuda available: True`.

## 3) Run Phase A workflow (exact order)

### 3.1 Train custom NumPy model (CPU)

```powershell
.\.venv\Scripts\python.exe train.py custom
```

### 3.2 Train PyTorch reference model (GPU)

```powershell
.\.venv\Scripts\python.exe train.py pytorch
```

### 3.3 Generate comparison artifacts

```powershell
.\.venv\Scripts\python.exe train.py compare
```

### 3.4 Generate analysis artifacts (Phase 6 visual outputs)

```powershell
.\.venv\Scripts\python.exe train.py analyze
```

## 4) Verify generated outputs

```powershell
Get-ChildItem .\results -File | Select-Object Name
Get-ChildItem .\results\latest\custom -File | Select-Object Name
Get-ChildItem .\results\latest\pytorch -File | Select-Object Name
```

You should have these top-level files in `results/` (at minimum):

- `comparison_train_loss.png`
- `comparison_val_loss.png`
- `comparison_accuracy_table.png`
- `comparison_training_curves.png`
- `confusion_matrix_custom.png`
- `confusion_matrix_pytorch.png`
- `per_class_accuracy_comparison.png`
- `sample_predictions_custom.png`
- `sample_predictions_pytorch.png`

You should also have latest run artifacts:

- `results/latest/custom/run_summary.json`
- `results/latest/custom/history.json`
- `results/latest/custom/metrics.json`
- `results/latest/custom/history.csv`
- `results/latest/custom/best_checkpoint.npz`
- `results/latest/pytorch/run_summary.json`
- `results/latest/pytorch/history.json`
- `results/latest/pytorch/metrics.json`
- `results/latest/pytorch/history.csv`
- `results/latest/pytorch/best_checkpoint.pt`

## 5) Pull exact numbers for README tables

```powershell
.\.venv\Scripts\python.exe -c "import json, pathlib; c=json.loads(pathlib.Path('results/latest/custom/run_summary.json').read_text()); p=json.loads(pathlib.Path('results/latest/pytorch/run_summary.json').read_text()); print('Custom test accuracy:', c['metrics']['test'].get('accuracy')); print('Custom test error:', c['metrics']['test'].get('misclassification_error')); print('PyTorch test accuracy:', p['metrics']['test'].get('accuracy')); print('PyTorch test error:', p['metrics']['test'].get('misclassification_error'))"
```

## 6) Optional: keep a full console log

If you want one log file for reproducibility:

```powershell
Start-Transcript -Path .\results\phase7_phaseA_run.log -Force
.\.venv\Scripts\python.exe train.py custom
.\.venv\Scripts\python.exe train.py pytorch
.\.venv\Scripts\python.exe train.py compare
.\.venv\Scripts\python.exe train.py analyze
Stop-Transcript
```

---

## Notes

- These commands use defaults from `NetworkConfig` (including 50 epochs and seed 9782).
- `custom` is NumPy-based and runs on CPU.
- `pytorch` will use CUDA automatically when available.
