"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np


class NeuralLayer:
    """
    Fully connected neural network layer
    """

    def __init__(self, input_size, output_size, activation=None, weight_init="xavier", weight_decay=0.0):
        """
        Initialize weights and biases
        
        Args:
            input_size: number of input neurons
            output_size: number of output neurons
            activation: activation function object
            weight_init: "random" or "xavier"
        """

        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        # Weight initialization
        if weight_init == "xavier":
            limit = np.sqrt(6 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        else:
            self.W = np.random.randn(input_size, output_size) * 0.01

        # Bias initialization
        self.b = np.zeros((1, output_size))
        self.weight_decay = weight_decay 

        # Store gradients
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        # Store input and output for backprop
        self.input = None
        self.z = None
        self.output = None


    def forward(self, X):
        """
        Forward propagation
        
        Z = XW + b
        A = activation(Z)
        """

        self.input = X

        self.z = np.dot(X, self.W) + self.b

        if self.activation is not None:
            self.output = self.activation.forward(self.z)
        else:
            self.output = self.z

        return self.output


    def backward(self, grad_output):
        """
        Backward propagation
        
        grad_output = dL/dA
        """

        # derivative of activation
        if self.activation is not None:
            grad_z = grad_output * self.activation.backward(self.z)
        else:
            grad_z = grad_output

        # gradients
        self.grad_W = np.dot(self.input.T, grad_z) + self.weight_decay * self.W
        self.grad_b = np.sum(grad_z, axis=0, keepdims=True)

        # gradient w.r.t input
        grad_input = np.dot(grad_z, self.W.T)

        return grad_input
