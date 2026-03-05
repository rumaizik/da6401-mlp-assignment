"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np


class CrossEntropyLoss:
    """
    Cross Entropy Loss for multi-class classification
    Used with Softmax output
    """

    def forward(self, y_true, y_pred):
        """
        Compute loss
        
        y_true: one-hot encoded true labels
        y_pred: predicted probabilities
        """
        self.y_true = y_true
        self.y_pred = y_pred

        # avoid log(0)
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)

        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

        return loss

    def backward(self):
        """
        Gradient of cross entropy with softmax
        """
        return (self.y_pred - self.y_true) / self.y_true.shape[0]


class MeanSquaredError:
    """
    Mean Squared Error Loss
    """

    def forward(self, y_true, y_pred):
        """
        Compute MSE loss
        """
        self.y_true = y_true
        self.y_pred = y_pred

        loss = np.mean((y_true - y_pred) ** 2)

        return loss

    def backward(self):
        """
        Gradient of MSE loss
        """
        return 2 * (self.y_pred - self.y_true) / self.y_true.shape[0]