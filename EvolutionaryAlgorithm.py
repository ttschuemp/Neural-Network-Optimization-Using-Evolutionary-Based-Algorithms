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
        
        
    def randomPop(self):
        NNlist = []
        for i in range(self.popSize):
            randomLayers = initializeParameters()
            sigma = 0.1
            mu = 0.5
            lr = abs(sigma * np.random.randn() + mu) # learning rate random from a normal distribution
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
        
        
        popParent = singleobjective(popParent, popOffSpring)

        return popParent
    
#    
#    
#    
#            if dominance[0] == True and dominance[1] == True: # if both true then its a dominated solution # np.greater() compares every element in the list and returns bool
#                #dominant solution
#                popParent.neuralNetworks[i] = popOffSpring.neuralNetworks[i] # replace parent by child
#            
#            if dominance[0] == True and dominance[1] == False:
#                popParent.neuralNetworks[i] = popOffSpring.neuralNetworks[i]
