"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
import numpy as np


class SGD:
    """
    Stochastic Gradient Descent
    """

    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, layer):
        layer.W -= self.lr * layer.grad_W
        layer.b -= self.lr * layer.grad_b


class Momentum:
    """
    Momentum Optimizer
    """

    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.v_W = {}
        self.v_b = {}

    def update(self, layer, layer_id):

        if layer_id not in self.v_W:
            self.v_W[layer_id] = np.zeros_like(layer.W)
            self.v_b[layer_id] = np.zeros_like(layer.b)

        self.v_W[layer_id] = self.momentum * self.v_W[layer_id] - self.lr * layer.grad_W
        self.v_b[layer_id] = self.momentum * self.v_b[layer_id] - self.lr * layer.grad_b

        layer.W += self.v_W[layer_id]
        layer.b += self.v_b[layer_id]


class Adam:
    """
    Adam Optimizer
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):

        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m_W = {}
        self.v_W = {}

        self.m_b = {}
        self.v_b = {}

        self.t = 0

    def update(self, layer, layer_id):

        if layer_id not in self.m_W:

            self.m_W[layer_id] = np.zeros_like(layer.W)
            self.v_W[layer_id] = np.zeros_like(layer.W)

            self.m_b[layer_id] = np.zeros_like(layer.b)
            self.v_b[layer_id] = np.zeros_like(layer.b)

        self.t += 1

        # Update moment estimates
        self.m_W[layer_id] = self.beta1 * self.m_W[layer_id] + (1 - self.beta1) * layer.grad_W
        self.v_W[layer_id] = self.beta2 * self.v_W[layer_id] + (1 - self.beta2) * (layer.grad_W ** 2)

        self.m_b[layer_id] = self.beta1 * self.m_b[layer_id] + (1 - self.beta1) * layer.grad_b
        self.v_b[layer_id] = self.beta2 * self.v_b[layer_id] + (1 - self.beta2) * (layer.grad_b ** 2)

        # Bias correction
        m_W_hat = self.m_W[layer_id] / (1 - self.beta1 ** self.t)
        v_W_hat = self.v_W[layer_id] / (1 - self.beta2 ** self.t)

        m_b_hat = self.m_b[layer_id] / (1 - self.beta1 ** self.t)
        v_b_hat = self.v_b[layer_id] / (1 - self.beta2 ** self.t)

        # Update weights
        layer.W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
        layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

class Nadam:

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m_W = {}
        self.v_W = {}

        self.m_b = {}
        self.v_b = {}

        self.t = 0


    def update(self, layer, layer_id):

        if layer_id not in self.m_W:

            self.m_W[layer_id] = np.zeros_like(layer.W)
            self.v_W[layer_id] = np.zeros_like(layer.W)

            self.m_b[layer_id] = np.zeros_like(layer.b)
            self.v_b[layer_id] = np.zeros_like(layer.b)

        self.t += 1

        gW = layer.grad_W
        gb = layer.grad_b

        # Update first moment
        self.m_W[layer_id] = self.beta1 * self.m_W[layer_id] + (1 - self.beta1) * gW
        self.m_b[layer_id] = self.beta1 * self.m_b[layer_id] + (1 - self.beta1) * gb

        # Update second moment
        self.v_W[layer_id] = self.beta2 * self.v_W[layer_id] + (1 - self.beta2) * (gW ** 2)
        self.v_b[layer_id] = self.beta2 * self.v_b[layer_id] + (1 - self.beta2) * (gb ** 2)

        # Bias correction
        mW_hat = self.m_W[layer_id] / (1 - self.beta1 ** self.t)
        mb_hat = self.m_b[layer_id] / (1 - self.beta1 ** self.t)

        vW_hat = self.v_W[layer_id] / (1 - self.beta2 ** self.t)
        vb_hat = self.v_b[layer_id] / (1 - self.beta2 ** self.t)

        # Nesterov correction
        mW_nesterov = self.beta1 * mW_hat + ((1 - self.beta1) * gW) / (1 - self.beta1 ** self.t)
        mb_nesterov = self.beta1 * mb_hat + ((1 - self.beta1) * gb) / (1 - self.beta1 ** self.t)

        # Update weights
        layer.W -= self.lr * mW_nesterov / (np.sqrt(vW_hat) + self.eps)
        layer.b -= self.lr * mb_nesterov / (np.sqrt(vb_hat) + self.eps)
        
class RMSProp:
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.v_W = {}
        self.v_b = {}

    def update(self, layer, layer_id):

        if layer_id not in self.v_W:
            self.v_W[layer_id] = np.zeros_like(layer.W)
            self.v_b[layer_id] = np.zeros_like(layer.b)

        self.v_W[layer_id] = self.beta * self.v_W[layer_id] + (1 - self.beta) * (layer.grad_W ** 2)
        self.v_b[layer_id] = self.beta * self.v_b[layer_id] + (1 - self.beta) * (layer.grad_b ** 2)

        layer.W -= self.lr * layer.grad_W / (np.sqrt(self.v_W[layer_id]) + self.epsilon)
        layer.b -= self.lr * layer.grad_b / (np.sqrt(self.v_b[layer_id]) + self.epsilon)

class NAG:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.v_W = {}
        self.v_b = {}

    def update(self, layer, layer_id):

        if layer_id not in self.v_W:
            self.v_W[layer_id] = np.zeros_like(layer.W)
            self.v_b[layer_id] = np.zeros_like(layer.b)

        v_W_prev = self.v_W[layer_id]
        v_b_prev = self.v_b[layer_id]

        self.v_W[layer_id] = self.momentum * self.v_W[layer_id] - self.lr * layer.grad_W
        self.v_b[layer_id] = self.momentum * self.v_b[layer_id] - self.lr * layer.grad_b

        layer.W += -self.momentum * v_W_prev + (1 + self.momentum) * self.v_W[layer_id]
        layer.b += -self.momentum * v_b_prev + (1 + self.momentum) * self.v_b[layer_id]    