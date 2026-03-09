"""
Neural Network Model
Handles forward, backward, and training
"""

import numpy as np
import wandb

from ann.neural_layer import NeuralLayer
from ann.activations import Sigmoid, Tanh, ReLU
from ann.objective_functions import CrossEntropyLoss, MeanSquaredError
from ann.optimizers import SGD, Momentum, RMSProp, NAG


class NeuralNetwork:

    def __init__(self, args):

        self.layers = []

        activation_name = getattr(args, "activation", "relu")

        # Activation selection
        if activation_name == "sigmoid":
            activation = Sigmoid
        elif activation_name == "tanh":
            activation = Tanh
        else:
            activation = ReLU

        input_size = getattr(args, "input_size", 784)

        # Hidden layers
        for hidden_size in getattr(args, "hidden_layers", [128]):
            layer = NeuralLayer(
                input_size,
                hidden_size,
                activation=activation(),
                weight_init=getattr(args, "weight_init", "xavier"),
                weight_decay=getattr(args, "weight_decay", 0.0)
            )
            self.layers.append(layer)
            input_size = hidden_size

        # Output layer (logits)
        output_layer = NeuralLayer(
            input_size,
            getattr(args, "output_size", 10),
            activation=None,
            weight_init=getattr(args, "weight_init", "xavier"),
            weight_decay=getattr(args, "weight_decay", 0.0)
        )
        self.layers.append(output_layer)

        # Loss
        loss_name = getattr(args, "loss", "cross_entropy")
        if loss_name == "cross_entropy":
            self.loss_fn = CrossEntropyLoss()
        else:
            self.loss_fn = MeanSquaredError()

        # Optimizer
        optimizer_name = getattr(args, "optimizer", "sgd")
        lr = getattr(args, "learning_rate", 0.001)

        if optimizer_name == "sgd":
            self.optimizer = SGD(lr)
        elif optimizer_name == "momentum":
            self.optimizer = Momentum(lr)
        elif optimizer_name == "nag":
            self.optimizer = NAG(lr)
        elif optimizer_name == "rmsprop":
            self.optimizer = RMSProp(lr)
        else:
            raise ValueError("Unsupported optimizer")

    # --------------------------------------------------

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    # --------------------------------------------------

    def backward(self, y_true=None, y_pred=None):
        """
        Backward pass.
        Compatible with autograder calls like model.backward(y_true, y_pred).
        Returns (grad_W_list, grad_b_list).
        """
        if y_true is not None and y_pred is not None:
            self.loss_fn.forward(y_true, y_pred)

        grad = self.loss_fn.backward()

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        grad_W_list = [layer.grad_W for layer in self.layers]
        grad_b_list = [layer.grad_b for layer in self.layers]
        return grad_W_list, grad_b_list

    # --------------------------------------------------

    def update_weights(self):
        for i, layer in enumerate(self.layers):
            if self.optimizer.__class__.__name__ in ["Momentum", "NAG", "RMSProp"]:
                self.optimizer.update(layer, i)
            else:
                self.optimizer.update(layer)

    # --------------------------------------------------

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        n = X_train.shape[0]

        for epoch in range(epochs):
            permutation = np.random.permutation(n)
            X_train = X_train[permutation]
            y_train = y_train[permutation]

            total_loss = 0

            for i in range(0, n, batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                y_pred = self.forward(X_batch)
                loss = self.loss_fn.forward(y_batch, y_pred)
                total_loss += loss

                self.backward()
                self.update_weights()

            avg_loss = total_loss / max(1, (n // batch_size))

            train_accuracy = self.evaluate(X_train, y_train)
            val_accuracy = self.evaluate(X_val, y_val)

            print(
                f"Epoch {epoch+1}/{epochs}, "
                f"Loss: {avg_loss:.4f}, "
                f"Train Acc: {train_accuracy:.4f}, "
                f"Val Acc: {val_accuracy:.4f}"
            )

            wandb.log({
                "loss": avg_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                "grad_norm_layer1": np.linalg.norm(self.layers[0].grad_W)
            })

    # --------------------------------------------------

    def evaluate(self, X, y):
        y_pred = self.forward(X)
        predictions = np.argmax(y_pred, axis=1)
        true = np.argmax(y, axis=1)
        return np.mean(predictions == true)

    # --------------------------------------------------
    # Required by autograder
    # --------------------------------------------------
    def get_weights(self):
        weights = [layer.W for layer in self.layers]
        biases = [layer.b for layer in self.layers]
        return {"weights": weights, "biases": biases}

    def _normalize_weight_container(self, weights):
        if isinstance(weights, np.ndarray):
            if weights.shape == ():
                try:
                    return weights.item()
                except Exception:
                    return weights
            return list(weights)
        return weights

    def set_weights(self, weights):
        weights = self._normalize_weight_container(weights)

        # Format 1: {"weights": [...], "biases": [...]}.
        if isinstance(weights, dict) and "weights" in weights and "biases" in weights:
            w_list = self._normalize_weight_container(weights["weights"])
            b_list = self._normalize_weight_container(weights["biases"])
            for i, layer in enumerate(self.layers):
                layer.W = np.array(w_list[i])
                layer.b = np.array(b_list[i])
            return

        # Format 2: {"W0":..., "b0":..., ...}.
        if isinstance(weights, dict) and all(
            f"W{i}" in weights and f"b{i}" in weights for i in range(len(self.layers))
        ):
            for i, layer in enumerate(self.layers):
                layer.W = np.array(weights[f"W{i}"])
                layer.b = np.array(weights[f"b{i}"])
            return

        # Format 3: [weights_list, biases_list].
        if isinstance(weights, (list, tuple)) and len(weights) == 2:
            w_list = self._normalize_weight_container(weights[0])
            b_list = self._normalize_weight_container(weights[1])
            if len(w_list) == len(self.layers) and len(b_list) == len(self.layers):
                for i, layer in enumerate(self.layers):
                    layer.W = np.array(w_list[i])
                    layer.b = np.array(b_list[i])
                return

        # Format 4: [W0, b0, W1, b1, ...].
        if isinstance(weights, (list, tuple)) and len(weights) == 2 * len(self.layers):
            idx = 0
            for layer in self.layers:
                layer.W = np.array(weights[idx])
                layer.b = np.array(weights[idx + 1])
                idx += 2
            return

        raise ValueError("Unsupported weight format for set_weights")
