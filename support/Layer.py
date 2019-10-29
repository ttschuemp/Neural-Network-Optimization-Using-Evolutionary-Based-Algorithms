# support/Layer.py

import numpy as np
from support.AbstractLayer import Layers

class Layer(Layers):

    def __init__(self, inputSize, outputSize):
        self.weights = np.random.randn(inputSize, outputSize) * 0.1 # * 0.01 keep initial weights small
        self.bias = np.random.randn(1, outputSize) * 0.1
        self.neurons = inputSize
        #Rprop 
        self.Nminus = 0.5# bounds form Rprop 0 < Nminus < 1 < Nplus
        self.Nplus = 1.2# bounds from Rprop
        self.deltaW = np.zeros((inputSize, outputSize)) + 0.1 # initial value of delta 
        self.deltaB = np.zeros((1, outputSize)) + 0.1 # initial value of delta 
        self.deltaMax = np.zeros((inputSize, outputSize))+ 50 
        self.deltaMin = np.zeros((inputSize, outputSize)) + 1e-6
        self.weightsError_old = np.random.uniform(-0.1, 0.1, size=(inputSize, outputSize)) # initial value
        self.outputError_old = np.random.uniform(-0.1, 0.1, size=(1, outputSize)) # initial value 
        self.delta_oldW = np.random.uniform(-0.1, 0.1, size=(inputSize, outputSize)) # initial value
        self.delta_oldB = np.random.uniform(-0.1, 0.1, size=(1, outputSize)) # initial value
        self.deltaMaxB = np.zeros((1, outputSize))+50 
        self.deltaMinB = np.zeros((1, outputSize)) + 1e-6
        # QuickPro
        self.weightsE_old = np.ones((inputSize, outputSize))-0.5
        self.dw_old = np.random.uniform(-0.1, 0.1, size=(1, outputSize))
        
        
    
    def changeSize(self, layer, inputSize, outputSize):
        layer.weights = np.random.randn(inputSize, outputSize) * 0.1
        layer.bias = np.random.randn(1, outputSize) * 0.1
        return layer
    
    def jitterWeights(self): # gaussian noise to each weight with prob 0.3, x dist-> N(0, 0.01)
        numrows, numcols = self.weights.shape
        prob = 0.3
        sigma = 0.01
        mu = 0
        index = np.random.rand(numrows, numcols) < prob # idex of jitter weights 
        noise = np.random.normal(mu, sigma, size = (numrows, numcols))
        noise[index==False] = 0.0 # only the weights that have been selected by prob
        self.weights = self.weights + noise
        
    def pruningWeights(self): # sets smallest 1% of weights to zero starting from input layer then iterates to outputlayer
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
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backwardPropagation(self, outputError, learningRate, learningAlgorithm):
        if(learningAlgorithm == "BP"):
            inputError = np.dot(outputError, self.weights.T) # outputerror weighted so see the impact of layer
            weightsError = np.dot(self.input.T, outputError)
            
            # update parameters
            self.weights = self.weights - learningRate * weightsError # lerningRate adjusts how much the change in weight is
            self.bias =self.bias - learningRate * outputError
            
            return inputError
#------------------------------------------------------------------------------

        elif(learningAlgorithm == "Rprop"): # Rprop-
            inputError = np.dot(outputError, self.weights.T) # outputerror weighted so see the impact of layer
            weightsError = np.dot(self.input.T, outputError)
#            r,d = self.weights.shape 
#            q,v = self.bias.shape
            
            # Weights - increase stepsize
            index = self.weightsError_old * weightsError > 0.0 # bool array to index
            X = np.minimum(self.Nplus * self.delta_oldW,self.deltaMax) # adjustment of stepsize 
            self.deltaW[index==True] = X[index==True]
            
            # Weights - decrease stepsize
            index = self.weightsError_old * weightsError < 0.0 # bool array to index
            X = np.maximum(self.Nminus * self.delta_oldW,self.deltaMin) # adjustment of stepsize 
            self.deltaW[index==True] = X[index==True]

            self.weights = self.weights - np.sign(weightsError) * self.deltaW # update weight
            
            # Bias
            index = self.outputError_old * outputError > 0.0
            X = np.minimum(self.Nplus * self.delta_oldB,self.deltaMaxB) # adjustment of stepsize 
            self.deltaB[index==True] = X[index ==True]
            
            index = self.outputError_old * outputError < 0.0
            X = np.maximum(self.Nminus * self.delta_oldB,self.deltaMinB) # adjustment of stepsize 
            self.deltaB[index==True] = X[index == True]
            
            self.bias = self.bias - np.sign(outputError) * self.deltaB # update bias
    
            self.weightsError_old = weightsError.copy() 
            self.outputError_old = outputError.copy()
            self.delta_oldW = self.deltaW.copy() # save old variable
            self.delta_oldB = self.deltaB.copy() # save old variable
            
            return inputError
        
        elif(learningAlgorithm == "Quickpro"):
        
            inputError = np.dot(outputError, self.weights.T) # outputerror weighted so see the impact of layer
            weightsError = np.dot(self.input.T, outputError) # dL/dw
            # avoid division by zero
            index = weightsError == self.weightsE_old
            self.weightsE_old[index==True] = 0.1
            
            dw = self.dw_old * (np.divide(weightsError,(np.subtract(self.weightsE_old, weightsError))))
            self.dw_old = dw.copy()
            self.weightsE_old = weightsError.copy() 
            
            # update parameters
            self.weights = self.weights - learningRate * dw 
            
            self.bias =self.bias - learningRate * outputError
            
            return inputError
        
        else:
           print("error")
        
        
        
