# support/Layer.py

import numpy as np
from support.AbstractLayer import Layers

class Layer(Layers):

    def __init__(self, inputSize, outputSize):
        self.weights = np.random.randn(inputSize, outputSize) * 0.01 # * 0.01 keep initial weights small
        self.bias = np.random.randn(1, outputSize) * 0.01
    
    def changeSize(self, layer, inputSize, outputSize):
        layer.weights = np.random.randn(inputSize, outputSize) * 0.01
        layer.bias = np.random.randn(1, outputSize) * 0.01
        return layer

    def forwardPropagation(self, inputData): # returns output for a give input
        self.input = inputData
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backwardPropagation(self, outputError, learningRate):
        inputError = np.dot(outputError, self.weights.T) # outputerror weighted so see the impact of layer
        weightsError = np.dot(self.input.T, outputError)
#        dBias = outputError

        # update parameters
        self.weights = self.weights - learningRate * weightsError # lerningRate adjusts how much the change in weight is
        self.bias =self.bias - learningRate * outputError
        return inputError

