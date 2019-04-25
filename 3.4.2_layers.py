import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def signal_propagation(X, W, B):
    A = np.dot(X, W) + B
    return A

def identity_fuynction(x):
    return x

## input to layer1
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
A1 = signal_propagation(X, W1, B1)
Z1 = sigmoid(A1)
## print(A1)
## print(Z1)

## layer1 to layer2
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
A2 = signal_propagation(Z1, W2, B2)
Z2 = sigmoid(A2)

## layer2 to output
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
A3 = signal_propagation(Z2, W3, B3)
Y = identity_fuynction(A3)

print(Y)