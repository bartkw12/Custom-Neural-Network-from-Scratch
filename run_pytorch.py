from custom_nn.config import default_config
from pytorch_nn import (
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