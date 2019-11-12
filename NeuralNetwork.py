# NeuralNetwork.py

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

from support.Functions import tanh, tanhDerivative, sigmoid, sigmoidDerivative, mse, \
mseDerivative, relu, reluDerivative, softmax, softmaxDerivative, crossEntropy, crossEntropyDerivative




class NeuralNetwork:
    #static variables 
    maxNeurons = 150
    minNeurons = 30
    maxHiddenLayers = 4
    sizeInput = 784
    sizeOutput = 10
    
    def __init__(self, layerList, loss = crossEntropy, lossDerivative = crossEntropyDerivative):
        self.layers = layerList
        self.loss = loss
        self.lossDerivative = lossDerivative
        self.mutations = []
        self.accuracyTrain = float('NAN')
        self.accuracyVali = float('NAN')
        self.accuracyTest = float('NAN')
        self.err = [] 
        self.prunedWeights = 0
        self.nrNeurons =  0
        self.trainingIterations = 0
        self.activationFunctions = []
         # variables for NSGAII
        self.solution = []
        self.ndominated = 0 
        self.crowdingDistance = float('NAN') 
        self.dominantRank = float('NAN')
        # variables for plotting
        self.nrNeurons_h = []


    def add(self,index, layer): # add layer to NN
        self.layers.insert(index,layer)


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
        return self.nrNeurons - self.prunedWeights
    
    def getAF(self): 
        n = 2
        i = 0
        self.activationFunctions = []
        for l in self.layers: # loop over every second element in list
            if i % n == 1:
                self.activationFunctions.append(l.activation.__name__)
            i += 1
        
        

    # predict output for given input
    def predict(self, inputs, target, testSample = False): # for predicting you need forwardPropagation only cause network already trained
        # sample dimension first
#        target = target[:,None] # fix second dimension of vector
        r,d = target.shape
        row ,col = inputs.shape
        scorecardVali = []
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
                scorecardVali.append(1)
            else: 
                scorecardVali.append(0)
        scorecard_arrayVali = np.asarray(scorecardVali)
        if testSample == True: 
            self.accuracyTest = scorecard_arrayVali.sum() /scorecard_arrayVali.size
        else:
            self.accuracyVali = scorecard_arrayVali.sum() /scorecard_arrayVali.size
    
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
    def train(self, xTrain, yTrain, epochs, minAcc = 1.0):
        self.nrNeurons =  self.getNrNeurons()
        self.nrNeurons_h.append(self.nrNeurons)
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
            self.trainingIterations += 1
            if self.accuracyTrain > minAcc:
                break

            
#            plt.plot(range(epochs), self.err, markersize=3)
#            plt.show()





