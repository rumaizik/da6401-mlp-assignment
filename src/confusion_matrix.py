import numpy as np
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork

wandb.init(project="da6401-mlp")

# Load dataset
X_train, y_train, X_test, y_test = load_data("mnist")

# Load saved model
model_data = np.load("model.npy", allow_pickle=True).item()

# Recreate model arguments
class Args:
    pass

args = Args()
args.input_size = X_test.shape[1]
args.output_size = y_test.shape[1]
args.hidden_layers = [128, 64]
args.activation = "relu"
args.loss = "cross_entropy"
args.optimizer = "adam"
args.learning_rate = 0.001
args.weight_init = "xavier"
args.weight_decay = 0.0

# Build model
model = NeuralNetwork(args)

# Load weights
for i, layer in enumerate(model.layers):
    layer.W = model_data["weights"][i]
    layer.b = model_data["biases"][i]

# Predict
y_pred = model.forward(X_test)

pred = np.argmax(y_pred, axis=1)
true = np.argmax(y_test, axis=1)

# Confusion matrix
cm = confusion_matrix(true, pred)

# Plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")

plt.title("Confusion Matrix (MNIST)")
plt.tight_layout()

# Log to W&B
wandb.log({"confusion_matrix": wandb.Image(plt)})