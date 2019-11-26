# Population.py 

import numpy as np 

from NeuralNetwork_Batch import NeuralNetwork
import copy # for making copies of objects 


class Population: 
    def __init__(self, neuralNetworks): 
        self.popSize = len(neuralNetworks)
        self.neuralNetworks = neuralNetworks
        self.averageAccTrain = float('NAN')
        self.averageAccVali = float('NAN')
        self.averageErr = float('NAN')
        self.averageNeurons = float('NAN')
        self.averageTop5_err = float('NAN')
        self.averageNeurons_h = []


    def add_NN(self, neuralNetwork): 
        self.neuralNetworks.append(neuralNetwork)

    def rmv_NN(self, neuralNetwork):
        self.neuralNetworks.remove(neuralNetwork)
        
    def copy_pop(self, population): 
        populationCopy = copy.deepcopy(population)
        return populationCopy
    
    def Top5average(self):
        a=[]
        for n in self.neuralNetworks:
            d=int(len(n.accuracyTest_h))
            a += n.accuracyTest_h
        a = np.asarray(a)
        a = np.reshape(a,(self.popSize,d))
        a_sorted = np.sort(a,axis =0)[::-1] 
        averageTop5 = np.mean(a_sorted[0:5,:], axis=0)
        return averageTop5
    
        
        
        
        
        
#    def evaluatePop(self): 
#        # in sample
#        totalTrain = 0.0
#        for n in self.neuralNetworks:
#            totalTrain += n.accuracyTrain
#            self.averageAccIS = totalTrain/self.popSize
#            pass
#        self.evaluateNrNeurons()
#        
#        # out of sample 
#        totalOOS = 0.0
#        totalErr = 0.0
#        for n in self.neuralNetworks: 
#            if n.dominantRank == 1:
#                self.elitestNN.append(n)
#                print("Best NN: ","IS: ", n.accuracyTrain, "OS: ", n.accuracyVali, "nrNeurons: ", n.nrNeurons)
#                pass
#            # average Accuracy over pop out of sample
#            totalOOS += n.accuracyOOS
#            self.averageAccVali = totalOOS/self.popSize
#            # average Error over pop.
#            totalErr += n.err[-1]
#            self.averageErr = totalErr/self.popSize
#            
#
#        return self.averageAccTrain, self.averageAccVali, self.averageErr, self.averageNeurons;
#    
    def evaluateNrNeurons(self): 
        totalNeurons = 0
        for n in self.neuralNetworks: 
            totalNeurons += n.getNrNeurons()
            self.averageNeurons = totalNeurons/self.popSize
            self.averageNeurons_h.append(self.averageNeurons)
            
    def printPop(self): 
        totalOOS = 0.0
        totalErr = 0.0
        totalTrain = 0.0
        self.evaluateNrNeurons()

        for n in self.neuralNetworks:
            totalOOS += n.accuracyVali
            self.averageAccVali = totalOOS/self.popSize

            
            # average Error over pop.
            totalErr += n.err
            self.averageErr = totalErr/self.popSize
            totalTrain += n.accuracyTrain
            self.averageAccTrain = totalTrain/self.popSize
            n.getAF()

            
        print('############# Print Population #################')
        print('Population Size: ',self.popSize)
        print('Av. Error: {:.2f}'.format(self.averageErr), 'Av. Neurons: {:.2f}'.format(self.averageNeurons))
        print('Av. Acc. Train: {:.2f}%'.format(self.averageAccTrain*100), 'Av. Acc. Vali: {:.2f}%'.format(100*self.averageAccVali))
        print('------------------------------------------------')
        i = 0
        for n in self.neuralNetworks:
            print('NN nr:', i, 'Training Iterations: ', n.trainingIterations)
            n.trainingIterations_h.append(n.trainingIterations)
            print('Acc. Training: {:.2f}%'.format(n.accuracyTrain*100), 'Acc. Vali: {:.2f}%'.format(n.accuracyVali*100))
            n.accuracyTrain_h.append(n.accuracyTrain)
            n.accuracyVali_h.append(n.accuracyVali)
            n.accuracyTest_h.append(n.accuracyTest)
            print('Mutations: ', n.mutations)
            print('Nr. Neurons: ', n.nrNeurons, 'Pruned Weights: ', n.prunedWeights)
            pweights = n.getNrPrunedWeights()
#            print('Connections: {:.1f}%'.format(100-pweights/(n.totalWeights)*100))
            print('AF: ', n.activationFunctions)
            print('Hidden Layers: ', int((len(n.layers)- 4) /2))
            print('------------------------------------------------')
            i +=1
        print('####################################################')
            
                
              

