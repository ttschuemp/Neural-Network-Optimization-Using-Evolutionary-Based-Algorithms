import pandas as pd
import numpy as np


from NeuralNetwork import NeuralNetwork
from support.Layer import Layer
from support.ActivationLayer import ActivationLayer
from support.Functions import tanh, tanhDerivative, sigmoid, sigmoidDerivative, mse, \
mseDerivative, relu, reluDerivative, softmax, softmaxDerivative, crossEntropy, crossEntropyDerivative
from support.Bootstrap import bootstrap
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")



# initialize NN
inputLayer = Layer(6,4)
activationFunction = ActivationLayer(tanh, tanhDerivative)
hiddenLayer = Layer(4, 4)
activationFunction2 = ActivationLayer(tanh, tanhDerivative)
#hiddenLayer2 = Layer(6, 4)
#activationFunction3 = ActivationLayer(tanh, tanhDerivative)
outputLayer = Layer(4, 2)
activationFunction4 = ActivationLayer(softmax, softmaxDerivative)


layerList = [inputLayer, activationFunction, hiddenLayer, activationFunction2, outputLayer, activationFunction4]

nn_simulation = NeuralNetwork(layerList, crossEntropy, crossEntropyDerivative)



# set parameters (weights)
np.random.seed(2)
nn_simulation.layers[0].weights = np.random.rand(6, 4) +1
nn_simulation.layers[0].bias = np.random.randn(1, 4) +1
nn_simulation.layers[2].weights = np.random.rand(4, 4) +1
nn_simulation.layers[2].bias = np.random.randn(1, 4) +1
nn_simulation.layers[4].weights = np.random.rand(4, 2) +1
nn_simulation.layers[4].bias = np.random.randn(1, 2) +1

# Draw N-samples
N = 1000 # number of samples
d = 6 # input-dimensions
m = 2 # output-dimension
mu = np.zeros(6)
sigma = np.eye(d)
X = np.random.multivariate_normal(mu, sigma, N)


# get Y from NN


Y = np.zeros((N,m))*np.nan
for i in range(N):
    output = X[i]
    for l in nn_simulation.layers:
        output = l.forwardPropagation(output)
    Y[i]= output



