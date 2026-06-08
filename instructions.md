# Running The PyTorch NN On A Dedicated GPU (3080 Ti)

Yes. You can run this PyTorch implementation on a 3080 Ti.

The important requirement is that your Python environment must have a CUDA-enabled PyTorch build installed. If that is true, the current code will automatically choose the GPU because `pytorch_nn.get_device()` prefers `cuda` over `mps` and `cpu`.

## 1. Open PowerShell In The Repo Root

```powershell
Set-Location "C:\Users\gts\Desktop\Platforms Sandbox\Use Case No37\Use Case No37 Testing\test\DL\Custom-Neural-Network-from-Scratch"
```

## 2. Create And Activate A Virtual Environment

If you already have `.venv`, only run the activation command.

```powershell
py -3.10 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python --version
```

## 3. Install Project Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 4. Check Whether Your Current PyTorch Install Sees The GPU

```powershell
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda device count:', torch.cuda.device_count()); print('device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

If that prints `cuda available: True`, you are ready to train on the 3080 Ti.

If that prints `False`, install a CUDA-enabled PyTorch build.

## 5. Install A CUDA-Enabled PyTorch Build If Needed

Run this only if Step 4 showed that CUDA is unavailable.

```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Then verify again:

```powershell
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

If PyTorch changes its recommended wheel URL in the future, use the selector on `https://pytorch.org/get-started/locally/` and replace the install command accordingly.

## 6. Run PyTorch Training And Test Evaluation

The current repository does not yet have a dedicated `python -m pytorch_nn` training entry point, so run the training flow with this PowerShell here-string:

```powershell
@'
from custom_nn.config import default_config
python -c "import json; from pathlib import Path; p = Path('results/pytorch_history.json'); print(p.exists()); print(json.load(p.open()).keys())"
    FashionMNISTNet,
    evaluate_test_set,
    get_device,
    prepare_dataloaders,
    save_history,
    train_model,
)

config = default_config()
device = get_device()

print(f"Using device: {device}")

train_loader, val_loader, test_loader = prepare_dataloaders(config)
model = FashionMNISTNet(config)

history = train_model(model, train_loader, val_loader, config=config, device=device)
history_path = save_history(history)
print(f"Saved PyTorch history to: {history_path}")

metrics = evaluate_test_set(model, test_loader, device=device)
print(metrics)
'@ | python -
```

What this does:

- loads the Fashion-MNIST data through the shared preprocessing pipeline
- builds the PyTorch network
- trains it using the current config values
- saves history to `results/pytorch_history.json`
- evaluates on the test set
- prints the final test misclassification error and returned metrics

## 7. Confirm That History Was Saved

```powershell
Get-ChildItem .\results
python -c "import json; from pathlib import Path; p = Path('results/pytorch_history.json'); print(p.exists()); print(json.load(p.open()) .keys())"
```

If you prefer, use this cleaner verification command instead:

```powershell
python -c "import json; from pathlib import Path; p = Path('results/pytorch_history.json'); data = json.load(p.open()); print(p.exists()); print(sorted(data.keys())); print(len(data['train_loss']), len(data['val_loss']))"
```

## 8. Optional: Generate The Custom NN History Too

If you want the comparison loader to work later, also run the original custom model once so it writes `results/custom_nn_history.json`:

```powershell
python train.py
```

## 9. Optional: Confirm Both History Files Load

```powershell
python -c "from pytorch_nn import load_comparison_histories; histories = load_comparison_histories(); print(sorted(histories.keys())); print(sorted(histories['custom_nn'].keys())); print(sorted(histories['pytorch_nn'].keys()))"
```

## 10. What To Expect On A 3080 Ti

- This workload is small enough to run on CPU, but the 3080 Ti is the better place to do real training runs.
- GPU training should be much faster than the VM if the VM does not expose a strong GPU.
- Dataset download happens the first time if Fashion-MNIST is not already cached.
- The current implementation saves training history, but it does not yet save a PyTorch model checkpoint automatically.

## 11. Quick Troubleshooting

If `torch.cuda.is_available()` is still `False`:

- confirm you are running inside the correct `.venv`
- confirm your NVIDIA driver is installed and current
- confirm the installed PyTorch build is a CUDA build, not CPU-only
- rerun the CUDA verification command from Step 4

If imports fail:

```powershell
pip install -e .
```

If you want to inspect which device the code will use:

```powershell
python -c "from pytorch_nn import get_device; print(get_device())"
```