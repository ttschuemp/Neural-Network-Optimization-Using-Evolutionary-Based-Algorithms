# support/AbstractLayer.py

from abc import ABC, abstractmethod


# Abstract Base Class
class Layers(ABC):

    @abstractmethod
    def __init__(self, inputSize, outputSize):
        self.input = None
        self.output = None
        pass

    @abstractmethod # abstract method must now be implemented in subclass
    def forwardPropagation(self, inputData):
        #
        # computes the output of a layer for a given input
        #
        pass

    @abstractmethod
    def backwardPropagation(self, outputError, learningRate, Rprop = False):
        #
        # computes the derivative of the error with respect to
        # its input (dE/dX) given derivative of the error with
        # respect to its output (dE/dY)
        # (dE/dX--> dE/dY for the layer before that one)
        pass
