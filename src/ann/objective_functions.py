"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np


class CrossEntropyLoss:

    def forward(self, y_true, logits):
        logits = np.array(logits)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        y_true = np.array(y_true)
        if y_true.ndim == 1:
            one_hot = np.zeros((y_true.shape[0], logits.shape[1]))
            one_hot[np.arange(y_true.shape[0]), y_true.astype(int)] = 1.0
            y_true = one_hot

        self.y_true = y_true

        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        epsilon = 1e-12
        probs_clipped = np.clip(self.probs, epsilon, 1.0 - epsilon)
        loss = -np.sum(y_true * np.log(probs_clipped)) / y_true.shape[0]
        return loss

    def backward(self):
        return (self.probs - self.y_true) / self.y_true.shape[0]


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
