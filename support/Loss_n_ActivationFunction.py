# support/Loss&ActivationFunction.py

import numpy as np

# activation functions

def tanh(x):
    return np.tanh(x);

def tanhDerivative(x):
    return 1-np.tanh(x)**2;

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidDerivative(x):
    S = sigmoid(x)
    return S * (1 - S)

# loss function and derivative

def mse(yTrue, yEst):
    return np.mean(np.power(yTrue-yEst, 2));

def mseDerivative(yTrue, yEst):
    return 2*(yEst-yTrue)/yTrue.size;
