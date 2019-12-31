# Experiment_ArtificialData.py

import numpy as np
import scipy.io as scio
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
from NeuralNetwork_Batch import NeuralNetwork
from support.Layer import Layer
from support.ActivationLayer import ActivationLayer
from support.Functions import tanh, tanhDerivative, sigmoid, sigmoidDerivative, mse, \
mseDerivative, relu, reluDerivative, softmax, softmaxDerivative, crossEntropy, crossEntropyDerivative
from support.Bootstrap import bootstrap, standardize, transformY, transformY_ad
from EvolutionaryAlgorithm import EvolutionaryAlgorithm
from NSGAII import NSGAII
from support.Plotting_helper import plot_objectives, plot_IterationSGD, plot_testAcc, plot_exploration, plot_AD



def artificialData_Exp():
    # load data
    X_train = np.load('/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/train_inputs.npy').T
    y_train_ = np.load('/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/train_targets.npy').T
    y_train = transformY_ad(y_train_,2)
    
    X_test = np.load('/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/test_inputs.npy').T
    y_test_ = np.load('/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/test_targets.npy').T
    y_test = transformY_ad(y_test_,2)
    
    # define neural network (with one hidden layer)
    
    input_D = 2
    H = 6
    output_D = 2
    
    inputLayer = Layer(input_D, H)
    af = ActivationLayer(relu, reluDerivative)
    outputLayer = Layer(H, output_D)
    af2 = ActivationLayer(softmax, softmaxDerivative)
    
    layerList = [inputLayer, af, outputLayer, af2]
    nn = NeuralNetwork(layerList, crossEntropy, crossEntropyDerivative)
    
    # train, test
    nn.train(X_train, y_train, epochs = 600) 
    pred = nn.predict(X_test, y_test, testSample = True)
    
    # simulate data with these trained parameters (true DGP -> hiddenlayer=1, neurons=10)
    d = 2
    n = 1500
    artificialData_X = (np.random.rand(n, d)-0.5)
    y = np.zeros((n,d)) * np.nan
    artificialData_y = nn.predict(artificialData_X, y)
    
    #TRAIN
    y_train = artificialData_y[0:1000,:]
    X_train = artificialData_X[0:1000,:]
    
    #VALIDATION
    y_vali = artificialData_y[1000:1250,:]
    X_vali = artificialData_X[1000:1250,:]
    
    #TEST
    y_test = artificialData_y[1250:,:]
    X_test = artificialData_X[1250:,:]
    
    # initialize 
    # in this experiment I fix the activation function to tanh otherwise I cant compare the coefficents that easy
    
    Populations = []
    EAs = []
    popSize = 18
    it = 10 # iterations
    minAcc = 0.9 # makes algorithm much faster!! and more fair comparison!
    
    for e in range(20): # 10 experiments
        
        NSGA = NSGAII()
        EA = EvolutionaryAlgorithm(xTrain = X_train, yTrain = y_train, 
                               popSize = popSize)
        initialPopulation = EA.randomPop(loss = crossEntropy, lossDerivative = crossEntropyDerivative) # random population
        # main loop
        population = initialPopulation
        for t in range(it):
            print('** GENERATION',t+1,'**')
            # reproduction and mutation
            offSpring = EA.makeOffspring(population, t-2)
            
            # train with Adam
            EA.trainPop(population, epochs = 3, minAcc = minAcc) 
            EA.trainPop(offSpring, epochs = 3, minAcc = minAcc) 
            minAcc += 0.02  
            
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
        
    
    plot_AD(Populations, it)
    plt.show()
    

if __name__ == "__main__":
    print("##########-##########-##########")
    print("Artificial Data Experiment")
    artificialData_Exp()
    print("##########-##########-##########")
          

