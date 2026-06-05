import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from custom_nn import NeuralNetwork, load_fashion_MNIST, preprocess_data
from custom_nn.config import default_config


def main():
    import matplotlib.pyplot as plt

    config = default_config()
    model = NeuralNetwork(config)

    train_dataset, test_dataset = load_fashion_MNIST(seed=config.seed)
    (X_train, Y_train), (X_validation, Y_validation), (X_test, Y_test) = preprocess_data(
        train_dataset,
        test_dataset,
    )

    history = model.train(X_train, Y_train, X_validation, Y_validation)
    history_path = Path(__file__).resolve().parent / "results" / "custom_nn_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)

    train_metrics = model.evaluate(X_train, Y_train)
    test_metrics = model.evaluate(X_test, Y_test)

    print(f"Final Training Misclassification Error: {100 * (1.0 - train_metrics['accuracy']):.2f} %")
    print(f"Final Test Misclassification Error: {100 * (1.0 - test_metrics['accuracy']):.2f} %")

    epochs_list = range(1, len(history["train_loss"]) + 1)
    plt.figure()
    plt.plot(epochs_list, history["train_loss"], label="Training Loss")
    plt.plot(epochs_list, history["val_loss"], label="Validation Loss")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Cross-entropy Loss")
    plt.title("Training and Validation Loss for Model I")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()