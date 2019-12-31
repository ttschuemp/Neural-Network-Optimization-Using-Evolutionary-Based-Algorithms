# NeuralNetwork_Batch.py

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
from support.Functions import tanh, tanhDerivative, sigmoid, sigmoidDerivative, mse, \
mseDerivative, relu, reluDerivative, softmax, softmaxDerivative, crossEntropy, crossEntropyDerivative


class NeuralNetwork:
    
    #static variables (need to be changed before running another experiment!)
#    # *** MNIST ***
#    maxNeurons = 600 
#    minNeurons = 150 
#    maxHiddenLayers = 6
#    sizeInput = 784 
#    sizeOutput = 10 
#    
     ## *** Wholesale ***
    maxNeurons = 40
    minNeurons = 8
    maxHiddenLayers = 3
    sizeInput = 6
    sizeOutput = 2
     
#    ## *** AD ***
#    maxNeurons = 20
#    minNeurons = 2 
#    maxHiddenLayers = 3
#    sizeInput = 2
#    sizeOutput = 2
    
    def __init__(self, layerList, loss = crossEntropy, lossDerivative = crossEntropyDerivative):
        self.layers = layerList
        self.loss = loss
        self.lossDerivative = lossDerivative
        self.mutations = []
        self.accuracyTrain = float('NAN')
        self.accuracyVali = float('NAN')
        self.accuracyTest = float('NAN')
        self.err = float('NAN')
        self.prunedWeights = 0
        self.nrNeurons =  0
        self.trainingIterations = 0
        self.activationFunctions = []
        self.totalWeights = 0
         # variables for NSGA-II
        self.solution = []
        self.ndominated = 0 
        self.crowdingDistance = float('NAN') 
        self.dominantRank = float('NAN')
        # variables for plotting (variables with _h are for plotting only)
        self.nrNeurons_h = []
        self.accuracyTrain_h = []
        self.accuracyVali_h= []
        self.activationFunctions_h = [] 
        self.accuracyTest_h= [] 
        self.err_h = []
        self.trainingIterations_h = []
        self.prunedWeights_h= []
        self.accuracyTrain_h_iteration = []
        self.activationFunctions_hh = []
        self.layers_h = []
        self.layers_h_generation = []
        self.accuracyTrain_h_batch = []


    def add(self,index, layer): # add layer to NN
        self.layers.insert(index,layer)

 
    def rmv(self, layer): # remove layer from NN
        self.layers.remove(layer)


    def getNrNeurons(self): # calculate nr. of neurons
        n = 2
        i = 0
        self.nrNeurons = 0
        self.layers_h.append((len(self.layers) /2 -2)) # number of hidden layers (-2 since one input one output)
        for l in self.layers: # loop over every second element in list since these are weight layers
            if i % n == 0: 
                self.nrNeurons += l.weights.shape[0]
            i += 1
        self.nrNeurons = self.nrNeurons + NeuralNetwork.sizeOutput
        return self.nrNeurons


    def getNrPrunedWeights(self): # calculate nr. of pruned weights
        n = 2
        i = 0
        self.prunedWeights = 0
        for l in self.layers: # loop over every second element in list
            if i % n == 0: 
                index = l.weights == 0
                summ = index.sum()
                self.prunedWeights += summ
            i += 1
        return self.prunedWeights


    def getAF(self): # get the activation functions of a NN
        n = 2
        i = 0
        self.activationFunctions = []
        af =[] 
        for l in self.layers: # loop over every second element in list 
            if i % n == 1:
                self.activationFunctions.append(l.activation.__name__)
                if l.activation.__name__ == 'tanh':
                    af.append(1)
                elif l.activation.__name__== 'sigmoid':
                    af.append(2)
                elif l.activation.__name__ == 'relu':
                    af.append(3)
            i += 1
        self.activationFunctions_h.append(af)
        self.activationFunctions_hh.append(self.activationFunctions)

    # predict output for given input
    def predict(self, inputs, target, testSample = False): # for predicting you need forwardPropagation only cause network already trained
        r,d = target.shape # get sample dimension
        row ,col = inputs.shape # get sample dimension
        scorecardVali = np.zeros(r)*np.nan # variable to get accuracy
        estimation = np.zeros((r,d))*np.nan
        # run network over all samples
        for i in range(0,r,self.batchSize):
            output = inputs[i:i+self.batchSize,:] # forward propagation a batch of samples
            for l in self.layers:
                output = l.forwardPropagation(output)
            estimation[i:i+self.batchSize]= output
            scorecardVali[i:i+self.batchSize] = np.argmax(target[i:i+self.batchSize,:], axis = 1) == np.argmax(output, axis = 1) #correct predicted or not
        if testSample == True: # get validation / test accuracy
            self.accuracyTest = scorecardVali.sum() /scorecardVali.size
        else:
            self.accuracyVali = scorecardVali.sum() /scorecardVali.size
    
        return estimation 


    def train(self, xTrain, yTrain, epochs, minAcc = 1.0, batchSize = 1): # train a NN and get training accuracy
        self.nrNeurons =  self.getNrNeurons() # for display 
        self.nrNeurons_h.append(self.nrNeurons) # for display 
        self.prunedWeights = self.getNrPrunedWeights() # for display 
        r, d = xTrain.shape # get sample dimension
        self.batchSize = batchSize
        # training loop
        for i in range(epochs):
            err = 0
            scorecardTrain = np.zeros(r)*np.nan # to get training accuracy
            t = 1
            for j in range(0,r,batchSize): # for all samples
                output = xTrain[j:j+batchSize,:] # forward propagation, a batch of samples 
                for l in self.layers: # loop over every layer in NN
                    output = l.forwardPropagation(output) # output for each row of the training data
                # compute loss only for display
                err = err + self.loss(yTrain[j:j+batchSize,:], output) # compare the output of each row with the target of this row/sample
                scorecardTrain[j:j+batchSize] = np.argmax(yTrain[j:j+batchSize,:], axis = 1) == np.argmax(output, axis = 1) 
                # backward propagation
                error = self.lossDerivative(yTrain[j:j+batchSize,:], output) # wholesale example output is 1x3 and target is 1x3
                for l in reversed(self.layers): # loop over every layer in NN
                    error = l.backwardPropagation(error)
                if t%10==0: # check every 10 batches if trained until minAcc
                    Acc=scorecardTrain[0:(batchSize*t-1)].sum()/(batchSize*t)
                    self.accuracyTrain_h_batch.append(Acc)
                    if Acc > minAcc:
                        break
                t+=1
            # calculate average error on all samples
            # for plotting only
            self.err /= r 
            self.err_h.append(err)
            self.accuracyTrain = scorecardTrain[0:(batchSize*t-1)].sum()/(batchSize*t)
            self.accuracyTrain_h_iteration.append(self.accuracyTrain)
            self.trainingIterations += 1
            if self.accuracyTrain > minAcc:
                break

            





