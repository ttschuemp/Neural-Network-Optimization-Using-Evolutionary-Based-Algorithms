# NeuralNetwork.py

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

from support.Functions import tanh, tanhDerivative, sigmoid, sigmoidDerivative, mse, \
mseDerivative, relu, reluDerivative, softmax, softmaxDerivative, crossEntropy, crossEntropyDerivative




class NeuralNetwork:
    #static variables 
    maxNeurons = 70
    minNeurons = 15
    maxHiddenLayers = 4
    sizeInput = 784
    sizeOutput = 10
    
    def __init__(self, layerList, loss = mse, lossDerivative = mseDerivative):
        self.layers = layerList
        self.loss = loss
        self.lossDerivative = lossDerivative
#        self.learningRate = learningRate
#        self.decreaseLR = decreaseLR
        self.accuracyTrain = float('NAN')
        self.accuracyOOS = float('NAN')
#        self.result = []
        self.err = []
        self.prunedWeights = 0
        self.nrNeurons =  self.getNrNeurons() - self.prunedWeights
         # variables for NSGAII
        self.solution = []
        self.ndominated = 0 
        self.crowdingDistance = float('NAN') 
        self.dominantRank = float('NAN') 

    def add(self, layer): # add layer to NN
        self.layers.append(layer)


    def rmv(self, layer):
        self.layers.remove(layer)

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
        

    # predict output for given input
    def predict(self, inputs, target): # for predicting you need forwardPropagation only cause network already trained
        # sample dimension first
#        target = target[:,None] # fix second dimension of vector
        r,d = target.shape
        row ,col = inputs.shape
        scorecardOOS = []
        estimation = np.zeros((r,d))*np.nan

        # run network over all samples
        for i in range(row):
            output = inputs[i] # forward propagation
#            output = output[:,None] # fix second dimension of vector
#            output = output.T
            for l in self.layers:
                output = l.forwardPropagation(output)
            estimation[i]= output
            if (np.argmax(output)== np.argmax(target[i])):
                scorecardOOS.append(1)
            else: 
                scorecardOOS.append(0)
        scorecard_arrayOOS = np.asarray(scorecardOOS)
        self.accuracyOOS = scorecard_arrayOOS.sum() /scorecard_arrayOOS.size
    
        return estimation 
#            print("YHat:", np.argmax(output))
#            self.result.append(output)
#            if (np.argmax(output)== np.argmax(targets[i])):
#                scorecardOOS.append(1)
#            else: 
#                scorecardOOS.append(0)
#                pass
#        scorecard_arrayOOS = np.asarray(scorecardOOS)
#        self.accuracyOOS = scorecard_arrayOOS.sum() /scorecard_arrayOOS.size
##        print ("accuracyOOS = ", self.accuracyOOS)


    # train the network
    def train(self, xTrain, yTrain, epochs):
        self.nrNeurons =  self.getNrNeurons()
        r, d = xTrain.shape # sample dimension first
        # training loop
        for i in range(epochs):
            err = 0
            scorecardTrain = []
            for j in range(r): # for all samples
                output = xTrain[j] # forward propagation, one sample 
                output = np.reshape(output,(1, d))
                for l in self.layers:
                    output = l.forwardPropagation(output) # output for each row of the training data

                # compute loss (for display purpose only)
                err = err + self.loss(yTrain[j], output) # compare the output of each row with the target of this row/sample
##                    print("error", self.err)
##                    print("Y:", np.argmax(yTrain[j]),"YHat:", np.argmax(output))
                if (np.argmax(yTrain[j]) == np.argmax(output)): # append correct or incorrect to list
#                     # network's answer matches correct answer, add 1 to scorecard
                     scorecardTrain.append(1)
                else:
                     # network's answer doesn't match correct answer, add 0 to scorecard
                     scorecardTrain.append(0)

                # backward propagation
                error = self.lossDerivative(yTrain[j], output) # wholesale example output is 1x3 and target is 1x3
#                    print("error BP:", error)
                for l in reversed(self.layers):     # Error is in example e 1x3 matrix/vector
                    error = l.backwardPropagation(error)
            # calculate average error on all samples
            err /= r
            self.err.append(err)
#                print("av error: ", err)
#            self.learningRate *= self.decreaseLR
            
            scorecard_arrayTrain = np.asarray(scorecardTrain)
            self.accuracyTrain = scorecard_arrayTrain.sum() /scorecard_arrayTrain.size
#            plt.plot(range(epochs), self.err, markersize=3)
#            plt.show()





