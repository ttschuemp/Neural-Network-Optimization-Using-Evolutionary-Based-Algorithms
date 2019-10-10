# NeuralNetwork.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from support.Layer import Layer
from support.Loss_n_ActivationFunction import tanh, tanhDerivative, sigmoid, sigmoidDerivative, mse, mseDerivative
from support.ActivationLayer import ActivationLayer




class NeuralNetwork:
    #static variables 
    maxNeurons = 450
    maxHiddenLayers = 4
    sizeInput = 784
    sizeOutput = 10
    
    def __init__(self, layerList, loss = mse, lossDerivative = mseDerivative, learningRate = 0.15):
        self.layers = layerList
        self.loss = loss
        self.lossDerivative = lossDerivative
        self.learningRate = learningRate
        self.accuracyIS = float('NAN')
        self.accuracyOOS = float('NAN')
        self.result = []
        self.err = float('NAN')
        self.nrNeurons =  self.getNrNeurons()
        self.solution = [] # variable for NSGAII
        self.ndominated = 0 # variable for NSGAII

    def add(self, layer): # add layer to NN
        self.layers.append(layer)


    def rmv(self, layer):
        self.layers.remove(layer)


    # predict output for given input
    def predict(self, inputTest, targetTest): # for predicting you need forwardPropagation only cause network already trained
        # sample dimension first
        length = len(inputTest)
        lengthRow = len(inputTest[0])
        scorecardOOS = []

        # run network over all samples
        for i in range(length):
            output = inputTest[i] # forward propagation
            output = np.reshape(output,(1, lengthRow))
            for l in self.layers:
                output = l.forwardPropagation(output)
#            print("YHat:", np.argmax(output))
            self.result.append(output)
            if (np.argmax(output)== np.argmax(targetTest[i])):
                scorecardOOS.append(1)
            else: 
                scorecardOOS.append(0)
                pass
        scorecard_arrayOOS = np.asarray(scorecardOOS)
        self.accuracyOOS = scorecard_arrayOOS.sum() /scorecard_arrayOOS.size
#        print ("accuracyOOS = ", self.accuracyOOS)


    # train the network
    def train(self, xTrain, yTrain, epochs):
            length = len(xTrain) # sample dimension first
            lengthRow = len(xTrain[0])
            # training loop
            for i in range(epochs):
                self.err = 0
                scorecardIS = []
                for j in range(length): # for all col.
                    output = xTrain[j] # forward propagation, one sample 
                    output = np.reshape(output,(1, lengthRow))
                    for l in self.layers:
                        output = l.forwardPropagation(output) # output for each row of the training data

                    # compute loss (for display purpose only)
                    self.err = self.err + self.loss(yTrain[j], output) # compare the output of each row with the target of this row/sample
#                    print("error", err)
#                    print("Y:", np.argmax(yTrain[j]),"YHat:", np.argmax(output))
                    if (np.argmax(yTrain[j]) == np.argmax(output)): # append correct or incorrect to list
                         # network's answer matches correct answer, add 1 to scorecard
                         scorecardIS.append(1)
                    else:
                         # network's answer doesn't match correct answer, add 0 to scorecard
                         scorecardIS.append(0)
                         pass
                    # backward propagation
                    error = self.lossDerivative(yTrain[j], output) # wholesale example output is 1x3 and target is 1x3
#                    print("error BP:", error)
                    for l in reversed(self.layers):     # Error is in example e 1x3 matrix/vector
                        error = l.backwardPropagation(error, self.learningRate)
                # calculate average error on all samples
                self.err /= length
#                print("av error: ", err)
                scorecard_arrayIS = np.asarray(scorecardIS)
                self.accuracyIS = scorecard_arrayIS.sum() /scorecard_arrayIS.size
#                print ("accuracyIS = ", self.accuracyIS)

    def getNrNeurons(self): #calculates nr of neurons without input and outputlayer, cause is anyway in every NN the same
        #         calculate nr. of neurons
        n = 2
        i = 0
        self.nrNeurons = 0
        for l in self.layers: # loop over every second element in list
            if i % n == 0 and i > 0: # i> 1 to skip the first layer
                self.nrNeurons += l.neurons
            i += 1
        return self.nrNeurons




