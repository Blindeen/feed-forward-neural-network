from abc import ABC, abstractmethod
import numpy as np

from activation_functions import LogisticFunction, ReluFunction, LeakyReluFunction


class Layer(ABC):
    """
    Abstract base class for a layer in a neural network.
    """
    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, grad):
        pass

    @abstractmethod
    def adjust(self, eta):
        pass

class Linear(Layer):
    """
    Linear layer with weights and biases.
    """
    def __init__(self, n_neurons, n_inputs):
        super().__init__()
        self.W = np.random.uniform(low=-0.1, high=0.1, size=(n_neurons, n_inputs))
        self.b = np.zeros((n_neurons,)) 

        self.X = None
        self.grad = None

    def forward(self, X):
        self.X = X
        return X @ self.W.T + self.b

    def backward(self, grad):
        self.grad = grad
        return grad @ self.W

    def adjust(self, eta):
        if self.grad.ndim == 1:
            self.W -= eta * self.grad.reshape(-1, 1) @ self.X.reshape(-1, 1).T
            self.b -= eta * self.grad
        else:
            batch_size = self.grad.shape[0]
            self.W -= eta * (self.grad.T @ self.X) / batch_size
            self.b -= eta * np.mean(self.grad, axis=0)

class Activation(Layer):
    """
    Activation layer that applies a specified activation function.
    """
    def __init__(self, activation):
        super().__init__()
        match activation:
            case 'logistic':
                self.activation = LogisticFunction()
            case 'relu':
                self.activation = ReluFunction()
            case 'leaky relu':
                self.activation = LeakyReluFunction()
            case _:
                raise Exception('Unknown activation function')
            
        self.X = None

    def forward(self, X):
        self.X = X
        return self.activation.value(X)

    def backward(self, grad):
        return grad * self.activation.derivative(self.X)

    def adjust(self, _):
        pass