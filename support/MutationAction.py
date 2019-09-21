# support/MutationAction.py

import numpy as np 

from support.Layer import Layer
from support.Parameters import randomActivationLayer

# switch like statement


def addLayer(neuralNetwork): # adds layer with XY neurons

    # new layer
    inputSize = np.random.randint(1,neuralNetwork.maxHiddenLayers+1) # random number for neurons of layer
    layerNew = Layer(inputSize, neuralNetwork.sizeOutput) # outputsize form static variable
    # new activation layer 
    activationLayer = randomActivationLayer()
    # adjust neighbor layer
    # inputSize of the new Layer is outputSize of old
    size = len(neuralNetwork.layers[-2].weights) # row length of weights
    neuralNetwork.layers[-2].changeSize(neuralNetwork.layers[-2], size, inputSize)
    neuralNetwork.add(layerNew)
    neuralNetwork.add(activationLayer)


def rmvLayer(neuralNetwork): # removes layer
    size = len(neuralNetwork.layers[-2].weights[0]) # col length
    neuralNetwork.rmv(neuralNetwork.layers[-2]) # rmv second last layer
    neuralNetwork.rmv(neuralNetwork.layers[-1])  # rmv last layer
    neuralNetwork.layers[-2].changeSize(neuralNetwork.layers[-2], size, neuralNetwork.sizeOutput)



def changeLr(neuralNetwork): # new and random learning rate
    neuralNetwork.learningRate = np.random.rand()


def jitterNN(neuralNetwork):
    n = 2 # iterate over every second element
    i = 0
    for l in neuralNetwork.layers: # loop over every second element in layer list cause these are
                                   # the layers with weights
        if i % n == 0:
            l.jitterWeights()
        i += 1



def five(direc):
    
    
    pass



def mutationAction(action, direc): # switch like function
    switcher = {
        1: addLayer,
        2: rmvLayer,
        3: changeLr,
        4: jitterNN,
        5: five,
    }
    # Get the function from switcher dictionary
    func = switcher.get(action, lambda: "Invalid")
    # Execute the function
    func(direc)
    
