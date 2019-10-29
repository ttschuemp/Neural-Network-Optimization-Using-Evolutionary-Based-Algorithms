# support/Parameters.py

import numpy as np


from support.Layer import Layer
from support.Loss_n_ActivationFunction import tanh, tanhDerivative, sigmoid, sigmoidDerivative, mse, mseDerivative
from support.ActivationLayer import ActivationLayer
from NeuralNetwork import NeuralNetwork



def randomActivationLayer(): # gives random a ActivationLayer with sigmoid or tanh
    b = np.random.randint(2) # toss a coin what activation layer
    if b == 0:
        activationLayer = ActivationLayer(sigmoid, sigmoidDerivative)
    else:
        activationLayer = ActivationLayer(tanh, tanhDerivative)
        pass
    return activationLayer


def initializeParameters(noHiddenLayers): # gives architecture of a NN
    # input layer
    x = np.random.randint(NeuralNetwork.minNeurons+1, (NeuralNetwork.maxNeurons+1)) # gives random int between 1 and maxNeurons
    inputLayer = Layer(NeuralNetwork.sizeInput, x)
    randLayerList = [inputLayer] # list that stores all layers
    activationLayerInput = randomActivationLayer()
    randLayerList.append(activationLayerInput)
    # hidden layers
    if(noHiddenLayers == False):
        hiddenLayers = np.random.randint(1, (NeuralNetwork.maxHiddenLayers+1)) # toss coin how many hidden layers; +1 cause of outputlayer
        s2 = np.random.randint(NeuralNetwork.minNeurons+1, NeuralNetwork.maxNeurons, size = hiddenLayers) # size of each hidden layer
        for i in range(hiddenLayers-1): #-1 cause outputlayer not in loop
            hiddenLayer = Layer(x, s2[i])
            x=s2[i]
            activationLayer = randomActivationLayer()
            randLayerList.append(hiddenLayer)
            randLayerList.append(activationLayer)
            pass
    # output layer
    outputLayer = Layer(x, NeuralNetwork.sizeOutput) #s2[-1] last element of array
    activationLayerOutput = randomActivationLayer()
    randLayerList.append(outputLayer)
    randLayerList.append(activationLayerOutput)

    return randLayerList



    
    