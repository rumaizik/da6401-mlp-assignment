"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np


class Sigmoid:
    """
    Sigmoid activation function
    """

    def forward(self, z):
        """
        σ(z) = 1 / (1 + e^-z)
        """
        self.output = 1 / (1 + np.exp(-z))
        return self.output

    def backward(self, z):
        """
        derivative: σ(z)(1 − σ(z))
        """
        sig = 1 / (1 + np.exp(-z))
        return sig * (1 - sig)


class Tanh:
    """
    Tanh activation function
    """

    def forward(self, z):
        """
        tanh(z)
        """
        self.output = np.tanh(z)
        return self.output

    def backward(self, z):
        """
        derivative: 1 − tanh²(z)
        """
        return 1 - np.tanh(z) ** 2


class ReLU:
    """
    ReLU activation function
    """

    def forward(self, z):
        """
        max(0, z)
        """
        self.output = np.maximum(0, z)
        return self.output

    def backward(self, z):
        """
        derivative: 1 if z > 0 else 0
        """
        return (z > 0).astype(float)


class Softmax:
    """
    Softmax activation function
    Used for output layer
    """

    def forward(self, z):
        """
        softmax(z) = exp(z) / sum(exp(z))
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        self.output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.output

    def backward(self, z):
        """
        Softmax gradient handled with cross-entropy,
        so this is usually not used directly
        """
        return np.ones_like(z)