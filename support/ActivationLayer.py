# support/ActivationLayer.py

from support.AbstractLayer import Layers

class ActivationLayer(Layers):

    def __init__(self, activation, activationDerivative):
        self.activation = activation
        self.activationDerivative = activationDerivative

    def forwardPropagation(self, inputData): # returns the activated input
        self.input = inputData
        self.output = self.activation(self.input)
        return self.output

    def backwardPropagation(self, outputError, learningRate): # learning Rate is not used!

        return self.activationDerivative(self.input) * outputError # here element multiplication
