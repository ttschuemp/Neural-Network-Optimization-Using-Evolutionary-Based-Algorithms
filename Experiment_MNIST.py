# Experiment_MNIST.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import sys
from NeuralNetwork_Batch import NeuralNetwork
from support.Layer import Layer
from support.ActivationLayer import ActivationLayer
from support.Functions import tanh, tanhDerivative, sigmoid, sigmoidDerivative, mse, \
mseDerivative, relu, reluDerivative, softmax, softmaxDerivative, crossEntropy, crossEntropyDerivative
from support.Bootstrap import bootstrap, standardize_image, transformY_mnist
from EvolutionaryAlgorithm import EvolutionaryAlgorithm
from NSGAII import NSGAII
from support.Plotting_helper import plot_objectives_2, plot_IterationSGD, plot_testAcc, plot_exploration


# Data 
dataPath_train = '/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/mnist_train (1).csv'
dataPath_test = '/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/mnist_test (1).csv'

def MNIST_EA():

    # load data
    data_train = pd.read_csv(dataPath_train, sep = ',', header = None, index_col = False)
    data_test = pd.read_csv(dataPath_test, sep = ',', header = None, index_col = False)
    data_train = data_train.iloc[0:10000,:]
    data_test = data_test.iloc[0:2000,:]
    
    # standardize data
    X = standardize_image(data_train.iloc[:,1:])
    data_train_standardized = pd.concat([data_train.iloc[:,0], X],axis = 1)
    X = standardize_image(data_test.iloc[:,1:])
    data_test_standardized = pd.concat([data_test.iloc[:,0], X],axis = 1)
    
    # bootstrap in train, vali and test
    _, data_vali, data_test = bootstrap(data_test_standardized, train = 0.0, validation = 0.5, test = 0.5, replacement = False) # default is 60/20/20
    data_train = data_train_standardized
    
    ## bootstrap in train, vali and test
    #data_train, data_vali, data_test = bootstrap(data_train_standardized, train = 0.6, validation = 0.2, test = 0.2, replacement = False) # default is 60/20/20
    
    # transform y and X in np.array 
    #TRAIN
    output = 10 # number of classes
    y_train = transformY_mnist(data_train, output)
    X_train = np.asanyarray(data_train.iloc[:,1:])
    
    #VALIDATION
    y_vali = transformY_mnist(data_vali, output)
    X_vali = np.asanyarray(data_vali.iloc[:,1:])
    
    #TEST
    y_test = transformY_mnist(data_test, output)
    X_test = np.asanyarray(data_test.iloc[:,1:])
    
    # initialize 
    
    Populations = []
    EAs = []
    
    for e in [6, 12, 18, 24]: 
        
        popSize = e  # e 
        it =10 # iterations
        minAcc = 0.90 # makes algorithm much faster!! and more fair comparison! Better convergence
        
        EA = EvolutionaryAlgorithm(xTrain = X_train, yTrain = y_train, 
                               popSize = popSize)
        NSGA = NSGAII()
        
        initialPopulation = EA.randomPop(loss = crossEntropy, lossDerivative = crossEntropyDerivative) # random population
        # main loop
        population = initialPopulation
        
        for t in range(it):
            print('** GENERATION',t+1,'**')
            # reproduction and mutation
            offSpring = EA.makeOffspring(population, t)
        
            # train with Adam
            EA.trainPop(population, epochs = 3, minAcc = minAcc, batchSize =10) 
            EA.trainPop(offSpring, epochs = 3, minAcc = minAcc, batchSize = 10) 
            minAcc += 0.01 
            
            # evaluate on validation dataset 
            EA.predPop(offSpring, X = X_vali, Y = y_vali)
            EA.predPop(population, X = X_vali, Y = y_vali)
         
            # selection
            newPopParent = NSGA.run(population, offSpring)
            # get the test accuracy (only for plotting)
            EA.predPop(newPopParent, X = X_test, Y = y_test, testSample = True)
            newPopParent.printPop()
        
            population = newPopParent
            
        Populations.append(population)
        EAs.append(EA)
        
    plot_objectives_2(Populations)
    plt.show()
        
    plot_IterationSGD(Populations)
    plt.show()
        
    plot_testAcc(Populations)
    plt.show()
            
    plot_exploration(EAs, it)
    plt.show()

if __name__ == "__main__":
    print("Python version in use: ", sys.version)
    print("Experiment_Wholesale")
    print("##########-##########-##########")
    MNIST_EA()
    print("##########-##########-##########")

