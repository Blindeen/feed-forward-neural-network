import numpy as np


class LogisticFunction:
    """
    Logistic activation function (also known as sigmoid).
    """
    def __init__(self, b=1):
        self.b = b
    
    def value(self, s):
        return 1 / (1 + np.exp(-self.b * s))
    
    def derivative(self, s):
        value = self.value(s)
        return self.b * value * (1 - value)
    
class ReluFunction:
    """
    Rectified Linear Unit (ReLU) activation function.
    """
    def value(self, s):
        return np.maximum(0, s)
    
    def derivative(self, s):
        if np.isscalar(s):
            return 1 if s > 0 else 0
        else:
            return np.where(s > 0, 1, 0)
    
class LeakyReluFunction: 
    """
    Leaky Rectified Linear Unit (Leaky ReLU) activation function.
    """
    def __init__(self, a=0.1):
        self.a = a
       
    def value(self, s):
        if np.isscalar(s):
            return s if s > 0 else self.a * s
        else:
            return np.where(s > 0, s, self.a * s)
    
    def derivative(self, s):
        if np.isscalar(s):
            return 1 if s > 0 else self.a
        else:
            return np.where(s > 0, 1, self.a)
