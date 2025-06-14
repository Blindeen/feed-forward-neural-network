import numpy as np
from sklearn import datasets, model_selection, preprocessing

from neural_network import NeuralNetworkClassifier
from utils import disp_cm

X, y = datasets.fetch_openml('mnist_784', return_X_y=True)
labels = np.unique(y)

X, y = X.to_numpy(), y.to_numpy(np.int8)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

scaler = preprocessing.MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

nn = NeuralNetworkClassifier(hidden_layer_sizes=(100, 50), activation='relu', max_iter=100, batch_size=100, lr=0.01, tol=1e-3)
nn.fit(X_train, y_train)
accuracy = nn.score(X_test, y_test)
print(f'Accuracy on MNIST dataset: {accuracy:.2f}')

y_pred = nn.predict(X_test)
disp_cm(y_test, y_pred, labels=labels)
