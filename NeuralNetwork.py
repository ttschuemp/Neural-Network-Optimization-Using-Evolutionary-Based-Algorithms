# NeuralNetwork.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from support.Layer import Layer
from support.Loss_n_ActivationFunction import tanh, tanhDerivative, sigmoid, sigmoidDerivative, mse, mseDerivative
from support.ActivationLayer import ActivationLayer
from support.BatchGenerator import BatchGenerator



class NeuralNetwork:
    #static variables 
    maxNeurons = 50
    maxHiddenLayers = 4
    sizeInput = 6
    sizeOutput = 3
    
    def __init__(self, layerList, loss = mse, lossDerivative = mseDerivative, learningRate = 0.05):
        self.layers = layerList
        self.loss = loss
        self.lossDerivative = lossDerivative
        self.learningRate = learningRate

    def add(self, layer): # add layer to NN
        self.layers.append(layer)

    def rmv(self, layer):
        self.layers.remove(layer)

    # predict output for given input
    def predict(self, inputData): # for predicting you need forwardPropagation only cause network already trained
        # sample dimension first
        length = len(inputData)
        result = []

        # run network over all samples
        for i in range(length):
            output = inputData[i] # forward propagation
            for layer in self.layers:
                output = layer.forwardPropagation(output)
            result.append(output)
        return result

    # train the network
    def train(self, xTrain, yTrain, epochs, batchSize = 0):
        if batchSize != 0:
            batch_generator = BatchGenerator(xTrain, yTrain, batchSize) # NEED EVERY EPOCH A NEW BATCH!!!!!
            xTrain, yTrain = batch_generator.next()
        else:
            length = len(xTrain) # sample dimension first
            lengthRow = len(xTrain[0])
            # training loop
            for i in range(epochs):
                err = 0
                scorecard = []
                for j in range(length):
                    output = xTrain[j] # forward propagation
                    output = np.reshape(output,(1, lengthRow))
                    for layer in self.layers:
                        output = layer.forwardPropagation(output) # output for each row of the training data

                    # compute loss (for display purpose only)
                    err = err + self.loss(yTrain[j], output) # compare the output of each row with the target of this row/sample
#                   print("Y:", np.argmax(yTrain[j]),"YHat:", np.argmax(output))
                    if (np.argmax(yTrain[j]) == np.argmax(output)): # append correct or incorrect to list
                         # network's answer matches correct answer, add 1 to scorecard
                         scorecard.append(1)
                    else:
                         # network's answer doesn't match correct answer, add 0 to scorecard
                         scorecard.append(0)
                         pass
                    # backward propagation
                    error = self.lossDerivative(yTrain[j], output) # wholesale example output is 1x3 and target is 1x3
                    for layer in reversed(self.layers):     # Error is in example e 1x3 matrix/vector
                        error = layer.backwardPropagation(error, self.learningRate)

                # calculate average error on all samples
                err /= length
                scorecard_array = np.asarray(scorecard)
                print ("performance = ", scorecard_array.sum() /scorecard_array.size)


    def evaluate(self): 
        
        pass



