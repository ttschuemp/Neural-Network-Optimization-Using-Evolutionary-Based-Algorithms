# support/Layer.py

import numpy as np
from support.AbstractLayer import Layers

class Layer(Layers): #weigth layer

    def __init__(self, inputSize, outputSize):
        self.weights = np.random.randn(inputSize, outputSize) * 0.1 # keep initial weights small
        self.bias = np.random.rand(1, outputSize) * 0.1 # np.random.rand(1, outputSize) *0.1
        # Adam
        self.alpha = 0.001 #Stepsize
        self.beta1 = 0.9 # Exponential decay rates for the moment estimates
        self.beta2 = 0.999 # Exponential decay rates for the moment estimates
        self.epsilon =1e-8
        self.m = 0 # first moment vector
        self.v = 0 # second moment vector
        self.mb = 0 # first moment vector (bais)
        self.vb = 0 # second moment vector (bais)
        self.t = 0 # time step

    def changeSize(self, layer, inputSize, outputSize): # change size of layer
        layer.weights = np.random.randn(inputSize, outputSize) * 0.1
        layer.bias = np.random.randn(1, outputSize) * 0.1

    def jitterWeights(self): # gaussian noise to each weight with prob 0.3, x dist-> N(0, 0.01)
        numrows, numcols = self.weights.shape
        prob = 0.3
        sigma = 0.2
        mu = 0
        index = np.random.rand(numrows, numcols) < prob # index of jitter weights
        noise = np.random.normal(mu, sigma, size = (numrows, numcols)) # noise matrix
        noise[index==False] = 0.0 # only the weights that have been selected by prob
        self.weights = self.weights + noise
        
    def pruningWeights(self): # sets smallest p% of weights to zero starting from input layer then iterates to outputlayer
        # excluding outputlayer until nrPruns is satisfied
        numrows, numcols = self.weights.shape
        absweights = abs(self.weights)
        min_= np.min(np.where(absweights==0, absweights.max(), absweights))
        index = min_== absweights
        zeros = np.zeros((numrows, numcols))
        self.weights[index==True] = zeros[index==True]
        return np.sum(index)
    
    def forwardPropagation(self, inputData): # returns output for a give input
        self.input = inputData
        self.output = np.dot(self.input, self.weights) + self.bias #dotproduct
        return self.output

    def backwardPropagation(self, outputError):
        
        inputError = np.dot(outputError, self.weights.T) 
        weightsError = np.dot(self.input.T, outputError)
        
        #Adam
        self.t = self.t+1
        # weights
        self.m = self.beta1 * self.m + (1 - self.beta1) * weightsError
        self.v = self.beta2 * self.v + (1 - self.beta2) * (weightsError**2)
        # bias
        self.mb = self.beta1 * self.mb + (1 - self.beta1) * outputError
        self.vb = self.beta2 * self.vb + (1 - self.beta2) * (outputError**2)
        # Bias-Correcton
        # weights
        mHat = self.m/(1 - self.beta1**self.t)
        vHat = self.v/(1 - self.beta2**self.t)
        # bias
        mHatb = self.mb/(1 - self.beta1**self.t)
        vHatb = self.vb/(1 - self.beta2**self.t)
        
        # update
        self.weights = self.weights - self.alpha * ((mHat)/(np.sqrt(vHat) - self.epsilon))
        self.bias = self.bias - self.alpha * ((mHatb)/(np.sqrt(vHatb) - self.epsilon))
        
        return inputError
    

