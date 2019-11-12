# Population.py 

import numpy as np 

from NeuralNetwork import NeuralNetwork
import copy # for making copies of objects 

class Population: 
    def __init__(self, neuralNetworks): 
        self.popSize = len(neuralNetworks)
        self.neuralNetworks = neuralNetworks
        self.averageAccTrain = float('NAN')
        self.averageAccVali = float('NAN')
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
                print("Best NN: ","IS: ", n.accuracyTrain, "OS: ", n.accuracyVali, "nrNeurons: ", n.nrNeurons)
                pass
            # average Accuracy over pop out of sample
            totalOOS += n.accuracyOOS
            self.averageAccVali = totalOOS/self.popSize
            # average Error over pop.
            totalErr += n.err[-1]
            self.averageErr = totalErr/self.popSize
            

        return self.averageAccTrain, self.averageAccVali, self.averageErr, self.averageNeurons;
    
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
            totalOOS += n.accuracyVali
            self.averageAccVali = totalOOS/self.popSize
            # average Error over pop.
            totalErr += n.err[-1]
            self.averageErr = totalErr/self.popSize
            totalTrain += n.accuracyTrain
            self.averageAccTrain = totalTrain/self.popSize
            n.getAF()
            
        print('############# Print Population #################')
        print('Population Size: ',self.popSize)
        print('Av. Error: {:.2f}'.format(self.averageErr[-1]), 'Av. Neurons: {:.2f}'.format(self.averageNeurons))
        print('Av. Acc. Train: {:.2f}%'.format(self.averageAccTrain*100), 'Av. Acc. Vali: {:.2f}%'.format(100*self.averageAccVali))
        print('------------------------------------------------')
        i = 0
        for n in self.neuralNetworks:
            print('NN nr:', i, 'Training Iterations: ', n.trainingIterations)
            print('Acc. Training: {:.2f}%'.format(n.accuracyTrain*100), 'Acc. Vali: {:.2f}%'.format(n.accuracyVali*100))
            print('Mutations: ', n.mutations)
            print('Nr. Neurons: ', n.nrNeurons, 'Pruned Neurons: ', n.prunedWeights)
            print('Connections: {:.1f}%'.format(100-n.prunedWeights/n.nrNeurons*100))
            print('AF: ', n.activationFunctions)
            print('Hidden Layers: ', int((len(n.layers)- 4) /2))
            print('------------------------------------------------')
            i +=1
        print('####################################################')
            
                
                

        
