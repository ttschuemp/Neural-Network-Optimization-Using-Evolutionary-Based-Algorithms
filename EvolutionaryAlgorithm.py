# EvolutionaryAlgorithm.py 

import numpy as np
import matplotlib.pyplot as plt

from Population import Population
from support.Parameters import initializeParameters
from support.MutationAction import mutationAction
from NeuralNetwork import NeuralNetwork
from support.evaluation_selection import singleobjective

# Initialization


class EvolutionaryAlgorithm:
    
    def __init__(self, epochs, xTrain, yTrain, popSize, xTest, yTest):
        self.epochs = epochs
        self.popSize = popSize
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.xTest = xTest
        self.yTest = yTest
        
        
    def randomPop(self, noHiddenLayers= False):
        NNlist = []
        for i in range(self.popSize):
            randomLayers = initializeParameters(noHiddenLayers=noHiddenLayers)
            sigma = 0.1
            mu = 0.5
            lr = abs(sigma * np.random.randn() + mu) # learning rate random from a normal distribution
            nn = NeuralNetwork(randomLayers, learningRate = lr)
            NNlist.append(nn)
            pass
        popNN = Population(NNlist)
        return popNN

    # Train Population
    def trainPop(self, population, learningAlgorithm): # popList gets a list of NN
        for n in population.neuralNetworks: # iterate over nn list 
             n.train(self.xTrain, self.yTrain, self.epochs, learningAlgorithm = learningAlgorithm)
             pass



    def makeOffspring(self, population): # returns offSpringPopulation
        
        # reproduction
        populationCopy = population.copy_pop(population)
        
        # mutation
        for n in populationCopy.neuralNetworks:
            mutationAction(1, n) # every child gets new layer
            x = np.random.poisson(1) 
            if(x==0):
                continue
            for i in range(x): # # mutation operations get executed with x+1; x = Poi(1)
                ranInt = np.random.randint(3)+2 # get random integers from 2-4
                mutationAction(ranInt, n) # manipulates a neural network 
                
        offSpringPopulation = populationCopy
        
        return offSpringPopulation
    
    
    def predPop(self, population): 
        for n in population.neuralNetworks:
            n.predict(self.xTest, self.yTest)
        pass
    


    def updatePop(self, popParent, popOffSpring): 
        # if child dominates parent then replace parent
        
        
        popParent = singleobjective(popParent, popOffSpring)

        return popParent




                