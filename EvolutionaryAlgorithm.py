# EvolutionaryAlgorithm.py 


import numpy as np

from Population import Population
from support.Parameters import initializeParameters
from support.MutationAction import mutationAction
from NeuralNetwork import NeuralNetwork

# Initialization


class EvolutionaryAlgorithm:
    
    def __init__(self, epochs, xTrain, yTrain, popSize, xTest, yTest):
        self.epochs = epochs
        self.popSize = popSize
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.xTest = xTest
        self.yTest = yTest
        
        
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
        for n in populationCopy.neuralNetworks:
            ranInt = np.random.randint(4)+1 # get random integers from 1-4
            mutationAction(ranInt, n) # manipulates a neural network 
            # add possibility that some networks get 1+x times mutated, where x is Poi(1) !!!!!!
        offSpringPopulation = populationCopy
        
        return offSpringPopulation
    
    
    def predPop(self, population): 
        for n in population.neuralNetworks:
            n.predict(self.xTest, self.yTest)
        pass
    
    
    
    def updatePop(self, popParent, popOffSpring): 
        # if child dominates parent then replace parent
        for i in range(popParent.popSize): 
            if popOffSpring.neuralNetworks[i].err < popParent.neuralNetworks[i].err:
                popParent.neuralNetworks[i] = popOffSpring.neuralNetworks[i] # replace parent by child
        
        
        
        return popParent
    
    
    
    

