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
        self.weight_decay = getattr(args, "weight_decay", 0.0)
        self.weight_init = getattr(args, "weight_init", "xavier")

        activation_name = getattr(args, "activation", "relu")

        if activation_name == "sigmoid":
            self.hidden_activation_cls = Sigmoid
        elif activation_name == "tanh":
            self.hidden_activation_cls = Tanh
        else:
            self.hidden_activation_cls = ReLU

        input_size = getattr(args, "input_size", 784)
        output_size = getattr(args, "output_size", 10)

        hidden_layers = getattr(args, "hidden_layers", None)
        if hidden_layers is None:
            hidden_size = getattr(args, "hidden_size", [128])
            num_layers = getattr(args, "num_layers", 1)
            if isinstance(hidden_size, int):
                hidden_layers = [hidden_size] * num_layers
            elif isinstance(hidden_size, (list, tuple)):
                hidden_layers = list(hidden_size)
                if len(hidden_layers) == 1 and num_layers > 1:
                    hidden_layers = hidden_layers * num_layers
            else:
                hidden_layers = [128]

        for hidden_size in hidden_layers:
            layer = NeuralLayer(
                input_size,
                hidden_size,
                activation=self.hidden_activation_cls(),
                weight_init=self.weight_init,
                weight_decay=self.weight_decay,
            )
            self.layers.append(layer)
            input_size = hidden_size

        output_layer = NeuralLayer(
            input_size,
            output_size,
            activation=None,
            weight_init=self.weight_init,
            weight_decay=self.weight_decay,
        )
        self.layers.append(output_layer)

        loss_name = getattr(args, "loss", "cross_entropy")
        if loss_name == "cross_entropy":
            self.loss_fn = CrossEntropyLoss()
        else:
            self.loss_fn = MeanSquaredError()

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

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def _is_output_like(self, arr):
        arr = np.array(arr)
        if arr.ndim == 1:
            return False
        return arr.shape[-1] == self.layers[-1].output_size

    def backward(self, arg1=None, arg2=None):
        """
        Supports:
        - backward() after loss_fn.forward(...)
        - backward(y_true, y_pred)
        - backward(X, y_true)

        Returns gradients from last layer to first:
        (grad_w_list, grad_b_list)
        """
        if arg1 is not None and arg2 is not None:
            a1_is_output = self._is_output_like(arg1)
            a2_is_output = self._is_output_like(arg2)

            if (not a1_is_output) and a2_is_output:
                # backward(X, y_true)
                X = np.array(arg1)
                y_true = np.array(arg2)
                y_pred = self.forward(X)
                self.loss_fn.forward(y_true, y_pred)
            elif a1_is_output and a2_is_output:
                # backward(y_true, y_pred)
                y_true = np.array(arg1)
                y_pred = np.array(arg2)
                self.loss_fn.forward(y_true, y_pred)
            else:
                # Fallback: treat as (y_true, y_pred)
                y_true = np.array(arg1)
                y_pred = np.array(arg2)
                self.loss_fn.forward(y_true, y_pred)

        grad = self.loss_fn.backward()

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        # Required order: last layer to first layer
        grad_W_list = [layer.grad_W for layer in reversed(self.layers)]
        grad_b_list = [layer.grad_b for layer in reversed(self.layers)]
        return grad_W_list, grad_b_list

    def update_weights(self):
        for i, layer in enumerate(self.layers):
            if self.optimizer.__class__.__name__ in ["Momentum", "NAG", "RMSProp"]:
                self.optimizer.update(layer, i)
            else:
                self.optimizer.update(layer)

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

            wandb.log(
                {
                    "loss": avg_loss,
                    "train_accuracy": train_accuracy,
                    "val_accuracy": val_accuracy,
                    "grad_norm_layer1": np.linalg.norm(self.layers[0].grad_W),
                }
            )

    def evaluate(self, X, y):
        y_pred = self.forward(X)
        predictions = np.argmax(y_pred, axis=1)
        true = np.argmax(y, axis=1)
        return np.mean(predictions == true)

    def get_weights(self):
        weights = [layer.W for layer in self.layers]
        biases = [layer.b for layer in self.layers]
        return {"weights": weights, "biases": biases}

    def _normalize_weight_container(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.shape == ():
                try:
                    return obj.item()
                except Exception:
                    return obj
            return list(obj)
        return obj

    def _extract_weight_lists(self, weights):
        weights = self._normalize_weight_container(weights)

        if isinstance(weights, dict) and "weights" in weights and "biases" in weights:
            w_list = self._normalize_weight_container(weights["weights"])
            b_list = self._normalize_weight_container(weights["biases"])
            return list(w_list), list(b_list)

        if isinstance(weights, dict):
            idx = 0
            w_list, b_list = [], []
            while f"W{idx}" in weights and f"b{idx}" in weights:
                w_list.append(weights[f"W{idx}"])
                b_list.append(weights[f"b{idx}"])
                idx += 1
            if idx > 0:
                return w_list, b_list

        if isinstance(weights, (list, tuple)) and len(weights) == 2:
            w_list = self._normalize_weight_container(weights[0])
            b_list = self._normalize_weight_container(weights[1])
            if isinstance(w_list, (list, tuple)) and isinstance(b_list, (list, tuple)):
                return list(w_list), list(b_list)

        if isinstance(weights, (list, tuple)) and len(weights) % 2 == 0 and len(weights) > 0:
            w_list, b_list = [], []
            for i in range(0, len(weights), 2):
                w_list.append(weights[i])
                b_list.append(weights[i + 1])
            return w_list, b_list

        raise ValueError("Unsupported weight format for set_weights")

    def _rebuild_layers_from_weights(self, w_list):
        self.layers = []
        for i, w in enumerate(w_list):
            w_arr = np.array(w)
            in_dim, out_dim = w_arr.shape
            activation = self.hidden_activation_cls() if i < len(w_list) - 1 else None
            layer = NeuralLayer(
                in_dim,
                out_dim,
                activation=activation,
                weight_init=self.weight_init,
                weight_decay=self.weight_decay,
            )
            self.layers.append(layer)

    def set_weights(self, weights):
        w_list, b_list = self._extract_weight_lists(weights)

        if len(w_list) != len(b_list):
            raise ValueError("Weights and biases length mismatch")

        need_rebuild = len(self.layers) != len(w_list)
        if not need_rebuild:
            for i, layer in enumerate(self.layers):
                if layer.W.shape != np.array(w_list[i]).shape or layer.b.shape != np.array(b_list[i]).shape:
                    need_rebuild = True
                    break

        if need_rebuild:
            self._rebuild_layers_from_weights(w_list)

        for i, layer in enumerate(self.layers):
            layer.W = np.array(w_list[i])
            layer.b = np.array(b_list[i])
            layer.grad_W = np.zeros_like(layer.W)
            layer.grad_b = np.zeros_like(layer.b)
