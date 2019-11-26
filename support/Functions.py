# support/Loss_n_ActivationFunction.py

import numpy as np

# Activation Function

def tanh(x):
    return np.tanh(x);

def tanhDerivative(x):
    t = np.tanh(x)
    return 1-np.power(t,2);

def sigmoid(x):
    # prevent overflow 
    x = np.clip(x, -600, 600 )
    return 1.0 / (1 + np.exp(-x))

def sigmoidDerivative(x):
    S = sigmoid(x)
    return S * (1 - S)

def softmax(x):
    x = np.clip(x, -600, 600 )
    return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1,1)

def softmaxDerivative(x):
    S = softmax(x)
    return S * (1 - S)

def relu(x):
    return (x > 0) * x # for x>0 = x, for x <= 0 = 0

def reluDerivative(x):
    return (x > 0) * 1 # for x>0 = 1, for x <= 0 = 0

# Loss Function 

def mse(y, yEst):
    return np.mean(np.power(y-yEst, 2));

def mseDerivative(y, yEst):
    m = y.shape[0]
    return 2*(yEst-y)/m;

def crossEntropy(y, yEst):
    eps = 1e-5 
    return  - np.sum(y*np.log(yEst+eps), axis =1)

def crossEntropyDerivative(y, yEst):
    m = y.shape[0]
    c = np.sum(yEst - y, axis = 0) 
    return c/m