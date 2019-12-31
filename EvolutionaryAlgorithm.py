# EvolutionaryAlgorithm.py 

import numpy as np
import matplotlib.pyplot as plt

from Population import Population
from support.Parameters import initializeParameters
from support.MutationAction import mutationAction
from NeuralNetwork_Batch import NeuralNetwork
from support.Evaluation_Selection import singleobjective


class EvolutionaryAlgorithm:
    
    def __init__(self, xTrain, yTrain, popSize):
        self.popSize = popSize
        self.xTrain = xTrain
        self.yTrain = yTrain
        # plotting variable
        self.exp_nrNeurons_h = []


    def randomPop(self, loss, lossDerivative): #Initializes random population of neural networks
        NNlist = []
        for i in range(self.popSize):
            randomLayers = initializeParameters() # function makes a random list of layers according to specification 
            nn = NeuralNetwork(randomLayers, loss= loss , lossDerivative = lossDerivative) # make instance of NN
            NNlist.append(nn)
        popNN = Population(NNlist)
        return popNN # outputs a population of NN

    # Train Population
    def trainPop(self, population, epochs, minAcc = 1.0, batchSize=1): # popList gets a list of NN
        for n in population.neuralNetworks: # iterate over nn list 
             n.train(self.xTrain, self.yTrain, epochs, minAcc, batchSize)
             self.exp_nrNeurons_h.append(n.nrNeurons)


    def makeOffspring(self, population, t): # returns offSpringPopulation # t is the generation
        # reproduction
        populationCopy = population.copy_pop(population)
        
        # mutation
        for n in populationCopy.neuralNetworks:
            x = np.random.poisson(1) 
            for i in range(x+1): # # mutation operations get executed with x+1; x = Poi(1)
                if t <= 5: # default is that mutation operations (1,2) get switched off in generation 5
                    # Mutation operation set 1
                    ranInt = np.random.randint(1,3) # get random integers from 1-2
                    # only mutation 1-2
                    mutationAction(ranInt, n) # manipulates a neural network
                # Mutation operation set 2
                ranInt = np.random.randint(3,6)
                # only mutation 3-5
                mutationAction(ranInt, n)
        offSpringPopulation = populationCopy
        return offSpringPopulation
    
    
    def predPop(self, population, X, Y, output = False, testSample = False): # predict population
        for n in population.neuralNetworks:
            pred=n.predict(inputs = X, target = Y, testSample = testSample)
            # suppresse output or not (accuracy gets calculated anyway in function NeuralNetwork.Predict() itselfe)
            if output == False:
                del pred
            else: 
                return pred

    #  Not used since we use NSGA-2, only as backup
    def updatePop(self, popParent, popOffSpring): 
        # if child dominates parent then replace parent
        popParent = singleobjective(popParent, popOffSpring)

        return popParent


