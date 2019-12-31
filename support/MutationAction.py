# support/MutationAction.py

import numpy as np 
import random
import math


from support.Layer import Layer
from support.Parameters import randomActivationLayer
from NeuralNetwork_Batch import NeuralNetwork

# switch like statement of mutation operation functions
def addLayer(neuralNetwork): # adds layer V_* with XY neurons to last layer {V_0 ,V_*, V_1, V_T} (V_0 input layer)
    
    # network already > max hidden layers ?
    if len(neuralNetwork.layers) >= (4 + (NeuralNetwork.maxHiddenLayers-1)*2): # cause 4 is a network with one hidden layer
        return
    else:
        # new layer
        outputSize = np.random.randint(NeuralNetwork.minNeurons+1, NeuralNetwork.maxNeurons+1) # random number for neurons of layer
        layerNew = Layer(NeuralNetwork.sizeInput, outputSize) # outputsize form static variable
        activationLayer = randomActivationLayer() # gives random activation layer
        # adjust neighbour layer
#        # inputSize of the new Layer is outputSize of old
        x,y = neuralNetwork.layers[0].weights.shape  # row length of weights
        neuralNetwork.layers[0].changeSize(neuralNetwork.layers[0], outputSize, y)
        
        # set values of Adam to 0 in changed layers
        neuralNetwork.layers[0].m = 0
        neuralNetwork.layers[0].mb = 0
        neuralNetwork.layers[0].v = 0
        neuralNetwork.layers[0].vb = 0
        neuralNetwork.layers[0].t = 0
        neuralNetwork.add(0,layerNew)
        
        # set values of Adam to 0 in changed layers
        neuralNetwork.layers[0].m = 0
        neuralNetwork.layers[0].mb = 0
        neuralNetwork.layers[0].v = 0
        neuralNetwork.layers[0].vb = 0
        neuralNetwork.layers[0].t = 0
        neuralNetwork.add(1, activationLayer)
        neuralNetwork.mutations.append('add layer') # mark that this operatin has been used
        
        
def rmvLayer(neuralNetwork): # removes random a hidden layer, hidden layers closer to input layer have higher prob to be rmved
    # Network to small ? one hidden layer is the minimal structure
    leng = len(neuralNetwork.layers)
    if leng <= 4: # 4 is min sturcture of network (single hidden layer)
        return
    else: 
        # not just the last layer remove, remove a random layer
        hiddenLayers = int((len(neuralNetwork.layers)-4)/2) # cause there are always 4 layers min of each network(inputL, outputL, 2xactivationL)
        # decide which hidden layer gets deleted
        x = np.array((range(hiddenLayers, 0, -1)))
        p = x/sum(x)
        cprob = np.cumsum(p)
        u = np.random.rand()
        i = len(np.where(u>cprob)[0]) # if 0 -> first hidden layer deleted (closer to input)
        i = i*2 + 2 #to get index of the layer
        o,z = neuralNetwork.layers[i].weights.shape
        neuralNetwork.rmv(neuralNetwork.layers[i]) # rmv layer
        neuralNetwork.rmv(neuralNetwork.layers[i-1]) # rmv activation layer
        # adjust size of other layers
        neuralNetwork.layers[i-2].changeSize(neuralNetwork.layers[i-2], neuralNetwork.layers[i-2].weights.shape[0] , z)
        # set values of Adam to 0 in changed layer
        neuralNetwork.layers[i-2].m = 0
        neuralNetwork.layers[i-2].mb = 0
        neuralNetwork.layers[i-2].v = 0
        neuralNetwork.layers[i-2].vb = 0
        neuralNetwork.layers[i-2].t = 0
        neuralNetwork.mutations.append('rmv layer') # mark that this operatin has been used


def jitterNN(neuralNetwork):
    n = 2
    i = 0
    for l in neuralNetwork.layers: # loop over every second element in layer list cause these are
                                   # the layers with weights
        if i % n == 0:
            l.jitterWeights() 
        i += 1
    neuralNetwork.mutations.append('jitter') # mark that this operatin has been used


def pruning(neuralNetwork): # delete the smalest prunFaktor % of weights exept for output layer
    nrNeurons = neuralNetwork.getNrNeurons()
    prunFaktor = 0.01 # 0.1 (wholesale)**** # 0.01 (MNIST)****
    nrPruns = math.ceil(nrNeurons * prunFaktor) # number of weights aimed to prune 
    n = 2 # iterate over every second element
    i = 0 
    deleted = 0
    while(nrPruns > deleted): 
        for l in neuralNetwork.layers: # loop over every second element in layer list cause these are
                                       # the layers with weights
            if i % n == 0 and l != neuralNetwork.layers[-2]: # exclude output layer
                deleted += l.pruningWeights() #starts with last layer until nrPruns is satisfied
            if (nrPruns <= deleted):
                neuralNetwork.mutations.append('pruning') # mark that this operatin has been used
                break
            i += 1
    
    
def changeAL(neuralNetwork): # change each hidden layers activation function with prob to random one of {sigmoid, tanh, relu}
    lenLayers = len(neuralNetwork.layers)
    prob = 0.5 #0.8 (wholesale)***** #0.5 (MNIST)***** # every random layer gets activation function changed with prob
    index = np.random.rand(1, lenLayers) < prob
    for i in range(0, lenLayers, 2):
        index[0,i] = False # exclude all weight layers
    index[0,-1] = False # exclude last activation layer
    if np.all(index == False): # if all index == False the operation will not applied so exit!
        return
    _,index = np.where(index == True) # gives indices of layers that get randomly changed
    for i in range(len(index)): 
        newAF = randomActivationLayer()
        del neuralNetwork.layers[index[i]]
        neuralNetwork.layers.insert(index[i], newAF)
    neuralNetwork.mutations.append('changeAL') # mark that this operatin has been used
        

def mutationAction(action, direc): # switch like function
    switcher = {
        1: addLayer,
        2: rmvLayer,
        3: jitterNN,
        4: pruning,
        5: changeAL, 
        }
    
    # Get the function from switcher dictionary
    func = switcher.get(action, lambda: "Invalid")
    # Execute the function
    func(direc)
    
