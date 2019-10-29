# support/MutationAction.py

import numpy as np 

from support.Layer import Layer
from support.Parameters import randomActivationLayer
from NeuralNetwork import NeuralNetwork

# switch like statement


def addLayer(neuralNetwork): # adds layer with XY neurons to last layer
    if len(neuralNetwork.layers) >= (4 + NeuralNetwork.maxHiddenLayers*2): # cause 4 is a network with no hidden layer
        pass
    else:
        # new layer
        inputSize = np.random.randint(NeuralNetwork.minNeurons+1, NeuralNetwork.maxNeurons+1) # random number for neurons of layer
        layerNew = Layer(inputSize, NeuralNetwork.sizeOutput) # outputsize form static variable
        # new activation layer 
        activationLayer = randomActivationLayer()
        # adjust neighbor layer
        # inputSize of the new Layer is outputSize of old
        size = len(neuralNetwork.layers[-2].weights) # row length of weights
        neuralNetwork.layers[-2].changeSize(neuralNetwork.layers[-2], size, inputSize)
        neuralNetwork.add(layerNew)
        neuralNetwork.add(activationLayer)


def rmvLayer(neuralNetwork): # removes random a hidden layer, hidden layers closer to input layer have higher prob to be rmved
    if len(neuralNetwork.layers) <= 4:
        pass
    else:
        pass  # not just the last layer remove, remove a random layer
#        hiddenLayers = (len(neuralNetwork.layers)-4)/2 # cause there are always 4 layers min of each network(inputL, outputL, 2xactivationL)
#        x = np.array((range(1,hiddenLayers)))
#        p = x/sum(x)
#        prob = np.cumsum(p)
#        u = np.random.rand()
#        i = len(np.where(u>prob)[0]) # hiddenlayer i will be deleted
#        i= i*2 #to get index of the layer
#        i=len(neuralNetwork.layers)-i
#        ----------------------------------
        size = len(neuralNetwork.layers[-4].weights) # col length, -4 cause this will be the last non-activation layer
        neuralNetwork.rmv(neuralNetwork.layers[-2]) # rmv second last layer
        neuralNetwork.rmv(neuralNetwork.layers[-1])  # rmv last layer
        neuralNetwork.layers[-2].changeSize(neuralNetwork.layers[-2], size, NeuralNetwork.sizeOutput)


# has no effect if larningAlgrithm is Rprop
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



def pruning(neuralNetwork): # delete the smalest 1 %  exept for output layer
    nrNeurons = neuralNetwork.getNrNeurons()
    prunFaktor = 0.01
    nrPruns = round(nrNeurons * prunFaktor)
    n = 2 # iterate over every second element
    i = 0 
    deleted = 0
    while(nrPruns > deleted): 
        for l in neuralNetwork.layers: # loop over every second element in layer list cause these are
                                       # the layers with weights
            if i % n == 0 and l != neuralNetwork.layers[-2]: # exclude output layer
                deleted += l.pruningWeights() #starts with last layer until nrPruns is satisfied
            if (nrPruns <= deleted):
                break
            i += 1
    neuralNetwork.prunedWeights += deleted
    
        

def mutationAction(action, direc): # switch like function
    switcher = {
        1: addLayer,
        2: pruning,
        3: changeLr,
        4: jitterNN,
        5: rmvLayer,

    }
    # Get the function from switcher dictionary
    func = switcher.get(action, lambda: "Invalid")
    # Execute the function
    func(direc)
    
