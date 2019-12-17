# support/Parameters.py

import numpy as np


from support.Layer import Layer
from support.Functions import tanh, tanhDerivative, sigmoid, sigmoidDerivative, mse, \
mseDerivative, relu, reluDerivative, softmax, softmaxDerivative, crossEntropy, crossEntropyDerivative
from support.ActivationLayer import ActivationLayer
from NeuralNetwork_Batch import NeuralNetwork



def randomActivationLayer(): # gives random a ActivationLayer with sigmoid or tanh
    b = np.random.randint(3) # 3
    if b == 0:
        activationLayer = ActivationLayer(sigmoid, sigmoidDerivative)
    elif b ==1:
        activationLayer = ActivationLayer(tanh, tanhDerivative)
    elif b==2:
        activationLayer = ActivationLayer(relu, reluDerivative)
    
    return activationLayer




def initializeParameters(): # gives architecture of a NN
    # input layer
    H = np.random.randint(NeuralNetwork.minNeurons+1, (NeuralNetwork.maxNeurons+1)) # gives random int between 1 and maxNeurons
    inputLayer = Layer(NeuralNetwork.sizeInput, H)
    randLayerList = [inputLayer] # list that stores all layers
    activationLayerInput = randomActivationLayer()
    randLayerList.append(activationLayerInput)
    # hidden layers
    hiddenLayers = np.random.randint(1, (NeuralNetwork.maxHiddenLayers)) # toss coin how many hidden layers
    H2 = np.random.randint(NeuralNetwork.minNeurons, NeuralNetwork.maxNeurons+1, size = hiddenLayers) # size of each hidden layer
    for i in range(hiddenLayers): #-1 cause outputlayer not in loop
        hiddenLayer = Layer(H, H2[i])
        H=H2[i]
        activationLayer = randomActivationLayer()
        randLayerList.append(hiddenLayer)
        randLayerList.append(activationLayer)
        
    # output layer
    outputLayer = Layer(H, NeuralNetwork.sizeOutput) 
    activationLayerOutput = ActivationLayer(softmax, softmaxDerivative)
    randLayerList.append(outputLayer)
    randLayerList.append(activationLayerOutput)

    return randLayerList



    
    