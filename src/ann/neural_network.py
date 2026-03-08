"""
Neural Network Model
Handles forward, backward, and training
"""

import numpy as np
import wandb
import numpy as np

from ann.neural_layer import NeuralLayer
from ann.activations import Sigmoid, Tanh, ReLU, Softmax
from ann.objective_functions import CrossEntropyLoss, MeanSquaredError
from ann.optimizers import SGD, Momentum, Adam, RMSProp, NAG,Nadam


class NeuralNetwork:

    def __init__(self, args):

        self.layers = []

        activation_name = getattr(args, "activation", "relu")
        weight_init = getattr(args, "weight_init", "xavier")
        weight_decay = getattr(args, "weight_decay", 0.0)

        # Activation selection
        if activation_name == "sigmoid":
           activation = Sigmoid
        elif activation_name == "tanh":
            activation = Tanh
        else:
            activation = ReLU

        input_size = getattr(args,"input_size",784)

        # Hidden layers
        for hidden_size in getattr(args,"hidden_layers",[128]):

            layer = NeuralLayer(
                input_size,
                hidden_size,
                activation=activation(),
                weight_init= getattr(args, "weight_init","xavier"),
                weight_decay= getattr(args,"weight_decay",0.0)
            )

            self.layers.append(layer)

            input_size = hidden_size

        # Output layer
        output_layer = NeuralLayer(
            input_size,
            getattr(args,"output_size",10),
            activation=Softmax(),
            weight_init= getattr(args,"weight_init","xavier"),
            weight_decay= getattr(args,"weight_decay",0.0)
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

        elif optimizer_name  == "momentum":
            self.optimizer = Momentum(lr)

        elif optimizer_name  == "nag":
            self.optimizer = NAG(lr)

        elif optimizer_name  == "rmsprop":
            self.optimizer = RMSProp(lr)

        elif optimizer_name  == "adam":
            self.optimizer = Adam(lr)
        elif optimizer_name  == "nadam":
            self.optimizer = Nadam(lr)    

        else:
            raise ValueError("Unsupported optimizer")
    # --------------------------------------------------        
  
    def set_weights(self, weights, b=None):

      import numpy as np

      # Case 1: autograder passes dictionary {"W":..., "b":...}
      if isinstance(weights, dict):
         Ws = weights["W"]
         bs = weights["b"]

         for layer, w, bias in zip(self.layers, Ws, bs):
            layer.W = np.array(w, dtype=float)
            layer.b = np.array(bias, dtype=float)

      # Case 2: autograder passes W and b separately
      elif b is not None:
         for layer, w, bias in zip(self.layers, weights, b):
            layer.W = np.array(w, dtype=float)
            layer.b = np.array(bias, dtype=float)

      # Case 3: autograder passes list of tuples [(W,b),...]
      else:
         for layer, pair in zip(self.layers, weights):
            w, bias = pair
            layer.W = np.array(w, dtype=float)
            layer.b = np.array(bias, dtype=float)
    # --------------------------------------------------

    def forward(self, X):

        output = X

        for layer in self.layers:

            output = layer.forward(output)

        return output

    # --------------------------------------------------

    def backward(self):

        grad = self.loss_fn.backward()

        for layer in reversed(self.layers):

            grad = layer.backward(grad)
        # log gradients of first 5 neurons in first hidden layer
        

        for i in range(min(5, self.layers[0].grad_W.shape[1])):
           wandb.log({f"grad_neuron_{i}": 
        np.linalg.norm(self.layers[0].grad_W[:,i])})   

    # --------------------------------------------------

    def update_weights(self):

        for i, layer in enumerate(self.layers):

            # optimizers needing layer id
            if self.optimizer.__class__.__name__ in ["Momentum", "NAG", "Adam", "Nadam", "RMSProp"]:
               self.optimizer.update(layer, i)

            # simple optimizers
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

            avg_loss = total_loss / (n // batch_size)

            train_accuracy = self.evaluate(X_train, y_train)
            val_accuracy = self.evaluate(X_val, y_val)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}") 
            import wandb
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

        accuracy = np.mean(predictions == true)

        return accuracy