# Hyperparameters for NN

# Training Parameters
BATCH_SIZE = 256
LEARNING_RATE = 0.002
EPOCHS = 50
LAMBDA = 0.001

# Model Parameters
INPUT_SIZE = 784
OUTPUT_SIZE = 10
HIDDEN_UNITS = 80
HIDDEN_LAYERS = 4

# Technique Parameters

# For Early Stopping
PATIENCE = 5
MIN_DELTA = 1e-5

# For Dropout
DROPOUT_RATE_INPUT = 0.10
DROPOUT_RATE_HIDDEN = 0.3

# For Batch Normalization
MOMENTUM = 0.185
EPSILON = 1e-5

# For ADAM optimizer
DECAY = 5e-7
EPSILON_A = 1e-7
BETA1 = 0.9
BETA2 = 0.999

# Other Settings
SEED = 9782