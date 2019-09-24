# Population.py 

import numpy as np 

from NeuralNetwork import NeuralNetwork
import copy # for making copies of objects 

class Population: 
    def __init__(self, neuralNetworks): 
        self.popSize = len(neuralNetworks)
        self.neuralNetworks = neuralNetworks
        self.averageAccIS = float('NAN')
        self.averageAccOOS = float('NAN')

    def add_NN(self, neuralNetwork): 
        self.neuralNetworks.append(neuralNetwork)

    def rmv_NN(self, neuralNetwork):
        self.neuralNetworks.remove(neuralNetwork)
        
    def copy_pop(self, population): 
        populationCopy = copy.deepcopy(population)
        return populationCopy
    
    def evaluatePop(self, ): 
        totalIS = 0.0
        for n in self.neuralNetworks:
            totalIS += n.accuracyIS
            self.averageAccIS = totalIS/self.popSize
            pass
        totalOOS = 0.0
        for n in self.neuralNetworks: 
            totalOOS += n.accuracyOOS
            self.averageAccOOS = totalOOS/self.popSize
            pass
        
        return self.averageAccIS, self.averageAccOOS;
        
