# Population.py 

import numpy as np 

from support.Parameters import makeRandomLayer 
from NeuralNetwork import NeuralNetwork
import copy # for making copies of objects 

class Population: 
    def __init__(self, neuralNetworks): 
        self.popSize = len(neuralNetworks)
        self.neuralNetworks = neuralNetworks

    def add_NN(self, neuralNetwork): 
        self.neuralNetworks.append(neuralNetwork)

    def rmv_NN(self, neuralNetwork):
        self.neuralNetworks.remove(neuralNetwork)
        
    def copy_pop(self, population): 
        populationCopy = copy.deepcopy(population)
        return populationCopy
