# support/Layer.py

import numpy as np
from support.AbstractLayer import Layers

class Layer(Layers):

    def __init__(self, inputSize, outputSize):
        self.weights = np.random.randn(inputSize, outputSize) * 0.1 # * 0.01 keep initial weights small
        self.bias = np.random.randn(1, outputSize) * 0.1
    
    def changeSize(self, layer, inputSize, outputSize):
        layer.weights = np.random.randn(inputSize, outputSize) * 0.1
        layer.bias = np.random.randn(1, outputSize) * 0.1
        return layer
    
    def jitterWeights(self):
        numrows = len(self.weights)    # 3 rows in your example
        numcols = len(self.weights[0]) # 2 columns in your example
        noise = (np.random.rand(numrows ,numcols) - 0.5) * 100 # 0.1 is the scalling for the nois; -0.5 so that also negativ values
        self.weights = self.weights + noise
    
    
    def forwardPropagation(self, inputData): # returns output for a give input
        self.input = inputData
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backwardPropagation(self, outputError, learningRate):
        inputError = np.dot(outputError, self.weights.T) # outputerror weighted so see the impact of layer
        weightsError = np.dot(self.input.T, outputError)
#       dBias = outputError
        # update parameters
#        print("old self.weighst:", self.weights)
        self.weights = self.weights - learningRate * weightsError # lerningRate adjusts how much the change in weight is
#        print("new self.weighst:", self.weights)
        self.bias =self.bias - learningRate * outputError
        return inputError


