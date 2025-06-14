import numpy as np
from layers import Linear, Activation


class NeuralNetworkClassifier:
    """
    A simple feedforward neural network classifier with backpropagation.
    Supports multiple hidden layers and various activation functions.
    """
    def __init__(self, hidden_layer_sizes=(100, ), activation='relu', max_iter=200, batch_size=100, lr=0.01, tol=1e-4):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_layer_activation = activation
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.lr = lr
        self.tol = tol
        
        self.layers = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_labels = np.unique(y).size
        self.__init_layers(n_features, n_labels)
        self.__learning(X, y, n_samples, n_labels)
        
    def predict(self, X):
        probabilities = self.__forward(X)
        return np.argmax(probabilities, axis=1)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def __init_layers(self, input_size, output_layer_size):
        current_input_size = input_size
        self.layers = []
        for size in self.hidden_layer_sizes:
            self.layers.append((Linear(size, current_input_size), Activation(self.hidden_layer_activation)))
            current_input_size = size
        self.layers.append((Linear(output_layer_size, current_input_size), Activation('logistic')))
    
    def __forward(self, X):
        for linear, activation in self.layers:
            X = linear.forward(X)
            X = activation.forward(X)
        
        return X
    
    def __backward(self, grad):
        reversed_layers = self.layers[::-1]
        for linear, activation in reversed_layers:
            grad = activation.backward(grad)
            grad = linear.backward(grad)

    def __adjust(self):
        for linear, _ in self.layers:
            linear.adjust(self.lr)

    def __learning(self, X, y, n_samples, n_labels):
        loss = np.inf
        n_batches = np.ceil(n_samples / self.batch_size)
        for _ in range(self.max_iter):
            for i in range(int(n_batches)):
                start = i * self.batch_size
                end = np.minimum((i + 1) * self.batch_size, n_samples)
                X_batch = X[start:end]
                y_batch = y[start:end]

                predictions = self.__forward(X_batch)
                targets = np.zeros((X_batch.shape[0], n_labels))
                targets[np.arange(X_batch.shape[0]), y_batch] = 1
                grad = predictions - targets
                self.__backward(grad)
                self.__adjust()

            all_predictions = self.__forward(X)
            all_targets = np.zeros((n_samples, n_labels))
            all_targets[np.arange(n_samples), y] = 1
            current_loss = np.sum((all_predictions - all_targets) ** 2)
            
            if np.abs(loss - current_loss) < self.tol:
                break
            loss = current_loss