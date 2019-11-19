# EvolutionaryAlgorithm.py 

import numpy as np
import matplotlib.pyplot as plt

from Population import Population
from support.Parameters import initializeParameters
from support.MutationAction import mutationAction
from NeuralNetwork import NeuralNetwork
from support.Evaluation_Selection import singleobjective



class EvolutionaryAlgorithm:
    
    def __init__(self, xTrain, yTrain, popSize):
        self.popSize = popSize
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.exp_nrNeurons_h = []
        
        
        
    def randomPop(self, loss, lossDerivative, noHiddenLayers= False):
        NNlist = []
        for i in range(self.popSize):
            randomLayers = initializeParameters(noHiddenLayers=noHiddenLayers)
            nn = NeuralNetwork(randomLayers, loss= loss , lossDerivative = lossDerivative)
            NNlist.append(nn)
            pass
        popNN = Population(NNlist)
        return popNN

    # Train Population
    def trainPop(self, population, epochs, minAcc = 1.0): # popList gets a list of NN
        for n in population.neuralNetworks: # iterate over nn list 
             n.train(self.xTrain, self.yTrain, epochs, minAcc)
             self.exp_nrNeurons_h.append(n.nrNeurons)
             n.nrNeurons_h.append(n.nrNeurons)
             



    def makeOffspring(self, population): # returns offSpringPopulation
        
        # reproduction
        populationCopy = population.copy_pop(population)
        
        # mutation
        for n in populationCopy.neuralNetworks:
            x = np.random.poisson(1) 
            for i in range(x+1): # # mutation operations get executed with x+1; x = Poi(1)
                ranInt = np.random.randint(1,3) # get random integers from 1-2
                # only mutation 1-2
                mutationAction(ranInt, n) # manipulates a neural network
                
                ranInt = np.random.randint(3,6)
                # only mutation 3-5
                mutationAction(ranInt, n)

                

                
        offSpringPopulation = populationCopy
        
        return offSpringPopulation
    
    
    def predPop(self, population, X, Y, output = False, testSample = False): 
        for n in population.neuralNetworks:
            pred=n.predict(inputs = X, target = Y, testSample = testSample)
            if output == False:
                del pred
            else: 
                return pred

        


    def updatePop(self, popParent, popOffSpring): 
        # if child dominates parent then replace parent
        
        
        popParent = singleobjective(popParent, popOffSpring)

        return popParent



                