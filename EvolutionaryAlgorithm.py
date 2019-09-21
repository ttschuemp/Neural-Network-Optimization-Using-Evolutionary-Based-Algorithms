# EvolutionaryAlgorithm.py 


import numpy as np

from Population import Population
from support.Parameters import initializeParameters
from support.MutationAction import mutationAction
from NeuralNetwork import NeuralNetwork

# Initialization


class EvolutionaryAlgorithm:
    
    def __init__(self, epochs, xTrain, yTrain, popSize, iterations):
        self.epochs = epochs
        self.iterations = iterations
        self.popSize = popSize
        self.xTrain = xTrain
        self.yTrain = yTrain
        
        
    def randomPop(self):
        NNlist = []
        for i in range(self.popSize):
            randomLayers = initializeParameters()
            lr = np.random.rand(1) # learning rate random
            nn = NeuralNetwork(randomLayers, learningRate = lr)
            NNlist.append(nn)
            pass
        popNN = Population(NNlist)
        return popNN

    # Train Population
    def trainPop(self, population): # popList gets a list of NN
        for n in population.neuralNetworks: # iterate over nn list 
             n.train(self.xTrain, self.yTrain, self.epochs)
             pass



    def makeOffspring(self, population): # returns offSpringPopulation
        
        # reproduction
        populationCopy = population.copy_pop(population)
        
        # mutation
        for n in populationCopy.neuralNetwork:
            
            mutationAction(2, n) # manipulates a neural network 
            # add possibility that some networks get 1+x times mutated, where x is Poi(1) !!!!!!
        offSpringPopulation = populationCopy
        
        return offSpringPopulation
    
    
    
    
    
    
    def updatePop(self): 
        
        return 
    
    
    
    

