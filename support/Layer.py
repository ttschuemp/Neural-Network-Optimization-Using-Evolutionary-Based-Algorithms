# support/Layer.py

import numpy as np
from support.AbstractLayer import Layers

class Layer(Layers):

    def __init__(self, inputSize, outputSize):
        self.weights = np.random.randn(inputSize, outputSize) * 0.1 # * 0.01 keep initial weights small
        self.bias = np.random.randn(1, outputSize) * 0.1
        self.neurons = inputSize
        
        self.Nminus = 0.5# bounds form Rprop 0 < Nminus < 1 < Nplus
        self.Nplus = 1.2# bounds from Rprop
        self.deltaW = np.zeros((inputSize, outputSize)) + 0.1 # initial value of delta 
        self.deltaB = np.zeros((1, outputSize)) + 0.1 # initial value of delta 
        self.deltaMax = np.zeros((inputSize, outputSize))+ 50 
        self.deltaMin = np.zeros((inputSize, outputSize)) + 1e-6
        self.weightsError_old = np.zeros((inputSize, outputSize)) # initial value
        self.outputError_old = np.zeros((1, outputSize)) 
        self.delta_oldW = np.zeros((inputSize, outputSize)) # initial value
        self.delta_oldB = np.zeros((1, outputSize))
        self.deltaMaxB = np.zeros((1, outputSize))+50 
        self.deltaMinB = np.zeros((1, outputSize)) + 1e-6
        
    
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

    def backwardPropagation(self, outputError, learningRate, Rprop = False):
        if(Rprop == False):
            inputError = np.dot(outputError, self.weights.T) # outputerror weighted so see the impact of layer
            weightsError = np.dot(self.input.T, outputError)
            
            # update parameters
            self.weights = self.weights - learningRate * weightsError # lerningRate adjusts how much the change in weight is
            self.bias =self.bias - learningRate * outputError
            
            return inputError
#------------------------------------------------------------------------------
        # MATRIX-Version
        else: # Rprop-
            inputError = np.dot(outputError, self.weights.T) # outputerror weighted so see the impact of layer
            weightsError = np.dot(self.input.T, outputError)
#            r,d = self.weights.shape # 2x3
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
    
            self.weightsError_old = weightsError 
            self.outputError_old = outputError
            self.delta_oldW = self.deltaW # save old variable
            self.delta_oldB = self.deltaB # save old variable
            
            return inputError
        
        
        
#                else: # Rprop-
#            inputError = np.dot(outputError, self.weights.T) # outputerror weighted so see the impact of layer
#            weightsError = np.dot(self.input.T, outputError)
#            r,d = self.weights.shape # 2x3
#            q,v = self.bias.shape
#            for i in range(r):  # 2
#                for j in range(d): # 3
#                    if self.weightsError_old[i,j] * weightsError[i,j] > 0.0: # larger stepsize
#                        self.deltaW[i,j] = min(self.Nplus * self.delta_oldW[i,j],self.deltaMax) # adjustment of stepsize, element wise minimum
#            
#                    elif self.weightsError_old[i,j] * weightsError[i,j] < 0.0: # jumpt over optimum
#                        self.deltaW[i,j] = max(self.Nminus * self.delta_oldW[i,j],self.deltaMin) # adjustment of stepsize 
#                    
#                    self.weights[i,j] = self.weights[i,j] - np.sign(weightsError[i,j]) * self.deltaW[i,j] # update weight
#                    
#                    if i == 0: 
#                        if self.outputError_old[i,j] * outputError[i,j] > 0.0:
#                            self.deltaB[i,j] = min(self.Nplus * self.delta_oldB[i,j],self.deltaMax) # adjustment of stepsize 
#
#                        elif self.outputError_old[i,j] * outputError[i,j] < 0.0:
#                            self.deltaB[i,j] = max(self.Nminus * self.delta_oldB[i,j],self.deltaMin)
#                            
#                        self.bias[i,j] = self.bias[i,j] - np.sign(outputError[i,j]) * self.deltaB[i,j] # update bias
#            
#            self.weightsError_old = weightsError 
#            self.delta_oldW = self.deltaW # save old variable
#            self.delta_oldB = self.deltaB # save old variable
#            
#            return inputError
#        
        
        
        
        
        
        
        
        
