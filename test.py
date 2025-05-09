from model import Layer_Dense, Activation_ReLU, Activation_Softmax, Categorical_Cross_entropy_loss
from techniques import Batch_Normalization, ADAM_Optimizer, Early_Stopping, Dropout
from data_preprocessing import load_fashion_MNIST, preprocess_data
from config import BATCH_SIZE, EPOCHS, LAMBDA, DROPOUT_RATE_INPUT, DROPOUT_RATE_HIDDEN, INPUT_SIZE, OUTPUT_SIZE, HIDDEN_UNITS, LEARNING_RATE, HIDDEN_LAYERS, MOMENTUM, EPSILON
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess data
train_dataset, test_dataset = load_fashion_MNIST()
(X_train, Y_train), (X_validation, Y_validation), (X_test, Y_test) = preprocess_data(train_dataset, test_dataset)

hidden_layers = HIDDEN_LAYERS

def build_model():
    layers = []

    # first layer block (set input size) Dense layer -> BN -> ReLU
    layers.append(Layer_Dense(n_inputs=INPUT_SIZE, n_neurons=HIDDEN_UNITS, l2_lambda=LAMBDA))
    layers.append(Batch_Normalization(n_neurons=HIDDEN_UNITS))
    layers.append(Activation_ReLU())

    # Dropout with diff rates for input vs. hidden layers
    # first call can use input_layer=True to get input dropout
    layers.append(Dropout(dropout_rate_input=DROPOUT_RATE_INPUT, dropout_rate_hidden=DROPOUT_RATE_HIDDEN))

    # Remaining hidden layers with regular dropout rate
    for _ in range(hidden_layers - 1):
        layers.append(Layer_Dense(n_inputs=HIDDEN_UNITS, n_neurons=HIDDEN_UNITS, l2_lambda=LAMBDA))
        layers.append(Batch_Normalization(n_neurons=HIDDEN_UNITS))
        layers.append(Activation_ReLU())
        # Dropout for hidden layers (second call onward, input_layer=False)
        layers.append(Dropout(dropout_rate_input=DROPOUT_RATE_INPUT, dropout_rate_hidden=DROPOUT_RATE_HIDDEN))

    # output layer using softmax and no dropout
    layers.append(Layer_Dense(n_inputs=HIDDEN_UNITS, n_neurons=OUTPUT_SIZE, l2_lambda=LAMBDA))
    layers.append(Activation_Softmax())

    return layers

layers = build_model()

# define objects
loss_function = Categorical_Cross_entropy_loss()
optimizer = ADAM_Optimizer(learning_rate=LEARNING_RATE)
early_stopper = Early_Stopping()


# forward pass call
def forward_pass(X, training=True):
    """
    Forward pass through all layers. 'training=True' ensures:
      - BN uses batch stats and updates running stats
      - Dropout is active
    'training=False':
      - BN uses running mean/variance
      - Dropout is disabled
    """
    # To manage input-layer dropout logic, track if dropout used once
    first_dropout_used = False

    # layer loop
    for layer in layers:
        if isinstance(layer, Batch_Normalization):
            X = layer.forward(X, training=training)
        elif isinstance(layer, Dropout):
            # First dropout -> input_layer=True, subsequent -> input_layer=False
            if not first_dropout_used:
                X = layer.forward(X, training=training, input_layer=True)
                first_dropout_used = True
            else:
                X = layer.forward(X, training=training, input_layer=False)
        else:
            # Dense, ReLU, Softmax
            X = layer.forward(X)
    return X

# backward pass logic
def backward_pass(y_true):
    """Backward pass from output to input layers."""
    dvalues = layers[-1].backward(y_true)  # Softmax backward
    for layer in reversed(layers[:-1]):
        if hasattr(layer, 'backward') and dvalues is not None:
            dvalues = layer.backward(dvalues)

# optimizer weight update
def update_weights():
    """Use Adam to update weights for Dense + BN."""
    optimizer.pre_update_params()

    for layer in layers:
        if isinstance(layer, Layer_Dense):
            optimizer.update_params(layer)
        elif isinstance(layer, Batch_Normalization):
            optimizer.update_params_bn(layer)  # Update gamma/beta with Adam

    optimizer.post_update_params()

# loss computation call
def compute_loss(X, Y, is_training=False):
    """
    For validation/test, set 'is_training=False' -> BN uses running stats, dropout off.
    """
    outputs = forward_pass(X, training=is_training)

    return loss_function.forward(outputs, Y)

def calculate_error(X, Y):
    """Compute 1-accuracy with BN & dropout off (training=False)."""
    outputs = forward_pass(X, training=False)
    predictions = np.argmax(outputs, axis=1)
    true_labels = np.argmax(Y, axis=1)
    accuracy = np.mean(predictions == true_labels)

    return 1.0 - accuracy

# -------------
# Training Loop test
# -------------

# Initialize lists to store losses for each epoch
training_losses = []
validation_losses = []

# training loop
for epoch in range(EPOCHS):
    # Shuffle training data again
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train, Y_train = X_train[indices], Y_train[indices]

    total_loss = 0.0
    num_batches = X_train.shape[0] // BATCH_SIZE

    for i in range(num_batches):
        start = i * BATCH_SIZE
        end = start + BATCH_SIZE
        X_batch = X_train[start:end]
        Y_batch = Y_train[start:end]

        # Forward pass (training=True -> BN uses batch stats, dropout is active)
        outputs = forward_pass(X_batch, training=True)
        batch_loss = loss_function.forward(outputs, Y_batch)
        total_loss += batch_loss

        # Backprop + weight update
        backward_pass(Y_batch)
        update_weights()

    # Average training loss
    avg_train_loss = total_loss / num_batches

    # Validation loss (BN + dropout off -> training=False)
    val_loss = compute_loss(X_validation, Y_validation, is_training=False)

    training_losses.append(avg_train_loss)
    validation_losses.append(val_loss)

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"  Training Loss:   {avg_train_loss:.4f}")
    print(f"  Validation Loss: {val_loss:.4f}")
    print("---------------------------------------")

    # Early stopping
    stop = early_stopper.forward(val_loss, layers)
    if stop:
        print(f"Early stopping triggered at epoch {epoch + 1}. Restoring best weights...")
        early_stopper.restore_best_weights(layers)
        break

# Final training & test misclassification error
final_train_error = calculate_error(X_train, Y_train)
final_test_error = calculate_error(X_test, Y_test)
print(f"\nFinal Training Misclassification Error: {100 * final_train_error:.2f} %")
print(f"Final Test Misclassification Error: {100 * final_test_error:.2f} %")

# plot the learning curves
epochs_list = range(1, len(training_losses) + 1)
plt.figure()
plt.plot(epochs_list, training_losses, label='Training Loss')
plt.plot(epochs_list, validation_losses, label='Validation Loss')
plt.xlabel('Number of Epochs')
plt.ylabel('Cross-entropy Loss')
plt.title('Training and Validation Loss for Model I')
plt.legend()
plt.show()