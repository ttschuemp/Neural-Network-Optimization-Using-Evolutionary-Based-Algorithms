# support/MutationAction.py

import numpy as np 
import random


from support.Layer import Layer
from support.Parameters import randomActivationLayer
from NeuralNetwork import NeuralNetwork

# switch like statement


def addLayer(neuralNetwork): # adds layer with XY neurons to last layer (inputlayer)
    
    # network already > max Hidden layers ?
    if len(neuralNetwork.layers) >= (4 + NeuralNetwork.maxHiddenLayers*2): # cause 4 is a network with no hidden layer
        return
#        ¨¨ranInt = np.random.randint(1,5) # random int 2-4
#        mutationAction(ranInt ,neuralNetwork)
        
    else:
        # new layer
        outputSize = np.random.randint(NeuralNetwork.minNeurons+1, NeuralNetwork.maxNeurons+1) # random number for neurons of layer
        layerNew = Layer(NeuralNetwork.sizeInput, outputSize) # outputsize form static variable
        activationLayer = randomActivationLayer() # gives random activation layer
        # adjust neighbor layer
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
        neuralNetwork.mutations.append('add layer')
        


def rmvLayer(neuralNetwork): # removes random a hidden layer, hidden layers closer to input layer have higher prob to be rmved
    # Network to small ? one hidden layer is the minimal structure
    leng = len(neuralNetwork.layers)
    if leng <= 6:
        return
#        ranInt = np.random.randint(1,5)
#        mutationAction(ranInt ,neuralNetwork)
        
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
        neuralNetwork.mutations.append('rmv layer')



def jitterNN(neuralNetwork):
    n = 2 # iterate over every second element
    i = 0
    for l in neuralNetwork.layers: # loop over every second element in layer list cause these are
                                   # the layers with weights
        if i % n == 0:
            l.jitterWeights()
        i += 1
    neuralNetwork.mutations.append('jitter')



def pruning(neuralNetwork): # delete the smalest 1 %  exept for output layer
    nrNeurons = neuralNetwork.getNrNeurons()
    prunFaktor = 0.01 # 0.1 (wholesale)****
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
                neuralNetwork.mutations.append('pruning')
                break
            i += 1
    
def changeAL(neuralNetwork):
    lenLayers = len(neuralNetwork.layers)
    prob = 0.5 #0.8 (wholesale)*****
    index = np.random.rand(1, lenLayers) < prob
    for i in range(0, lenLayers, 2):
        index[0,i] = False # exclude all weight layers
    index[0,-1] = False # exclude last activation layer
    if np.all(index == False):
        return
    _,index = np.where(index == True) # gives indices of layers that get randomly changed
    for i in range(len(index)): 
        newAF = randomActivationLayer()
        del neuralNetwork.layers[index[i]]
        neuralNetwork.layers.insert(index[i], newAF)
    neuralNetwork.mutations.append('changeAL')
        

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
    
