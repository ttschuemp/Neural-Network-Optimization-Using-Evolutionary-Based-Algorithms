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
        self.elitestNN = []
        self.averageErr = float('NAN')
        self.averageNeurons = float('NAN')

    def add_NN(self, neuralNetwork): 
        self.neuralNetworks.append(neuralNetwork)

    def rmv_NN(self, neuralNetwork):
        self.neuralNetworks.remove(neuralNetwork)
        
    def copy_pop(self, population): 
        populationCopy = copy.deepcopy(population)
        return populationCopy
        
    def evaluatePop(self): 
        # in sample
        totalIS = 0.0
        for n in self.neuralNetworks:
            totalIS += n.accuracyIS
            self.averageAccIS = totalIS/self.popSize
            pass
        
        # out of sample 
        totalOOS = 0.0
        totalErr = 0.0
        elitesErr = 99999 # just a high number for initialization
        for n in self.neuralNetworks: 
            if n.err < elitesErr:
                self.elitestNN = n
                elitesErr = n.err
                pass
            # average Accuracy over pop out of sample
            totalOOS += n.accuracyOOS
            self.averageAccOOS = totalOOS/self.popSize
            # average Error over pop.
            totalErr += n.err
            self.averageErr = totalErr/self.popSize
            

        return self.averageAccIS, self.averageAccOOS, self.averageErr;
    
    def evaluateNrNeurons(self): 
        totalNeurons = 0
        for n in self.neuralNetworks: 
            totalNeurons += n.getNrNeurons()
            self.averageNeurons = totalNeurons/self.popSize
            
            

        
