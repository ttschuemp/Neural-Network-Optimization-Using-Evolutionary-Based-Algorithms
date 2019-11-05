# Population.py 

import numpy as np 

from NeuralNetwork import NeuralNetwork
import copy # for making copies of objects 

class Population: 
    def __init__(self, neuralNetworks): 
        self.popSize = len(neuralNetworks)
        self.neuralNetworks = neuralNetworks
        self.averageAccTrain = float('NAN')
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
        totalTrain = 0.0
        for n in self.neuralNetworks:
            totalTrain += n.accuracyTrain
            self.averageAccIS = totalTrain/self.popSize
            pass
        self.evaluateNrNeurons()
        
        # out of sample 
        totalOOS = 0.0
        totalErr = 0.0
        for n in self.neuralNetworks: 
            if n.dominantRank == 1:
                self.elitestNN.append(n)
                print("Best NN: ","IS: ", n.accuracyTrain, "OS: ", n.accuracyOOS, "nrNeurons: ", n.nrNeurons)
                pass
            # average Accuracy over pop out of sample
            totalOOS += n.accuracyOOS
            self.averageAccOOS = totalOOS/self.popSize
            # average Error over pop.
            totalErr += n.err[-1]
            self.averageErr = totalErr/self.popSize
            

        return self.averageAccTrain, self.averageAccOOS, self.averageErr, self.averageNeurons;
    
    def evaluateNrNeurons(self): 
        totalNeurons = 0
        for n in self.neuralNetworks: 
            totalNeurons += n.getNrNeurons()
            self.averageNeurons = totalNeurons/self.popSize
            
    def printPop(self): 
        totalOOS = 0.0
        totalErr = 0.0
        totalTrain = 0.0
        self.evaluateNrNeurons()
        for n in self.neuralNetworks:
            totalOOS += n.accuracyOOS
            self.averageAccOOS = totalOOS/self.popSize
            # average Error over pop.
            totalErr += n.err[-1]
            self.averageErr = totalErr/self.popSize
            totalTrain += n.accuracyTrain
            self.averageAccTrain = totalTrain/self.popSize
            
        print('############# Print Population #################')
        print('Population Size: ', self.popSize)
        print('Av. Error: ', self.averageErr[-1], 'Av. Neurons: ',self.averageNeurons,)
        print('Av. Acc. Train: ',self.averageAccTrain, 'Av. Acc. Vali: ', self.averageAccOOS)
        print('------------------------------------------------')
        i = 0
        for n in self.neuralNetworks:
            print('NN nr:', i)
            print('Acc. Training: ', n.accuracyTrain, 'Acc. Vali: ', n.accuracyOOS)
            print('Mutations: ', n.mutations)
            print('Nr. Neurons: ', n.nrNeurons, 'Pruned Neurons: ', n.prunedWeights)
            print('Hidden Layers: ', (len(n.layers)- 4) /2)
            print('------------------------------------------------')
            i +=1
        print('####################################################')
            
                
                

        
