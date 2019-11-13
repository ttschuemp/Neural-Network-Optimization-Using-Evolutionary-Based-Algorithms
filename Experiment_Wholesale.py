# Experiment_Wholesale.py

import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

from NeuralNetwork import NeuralNetwork
from support.Layer import Layer
from support.ActivationLayer import ActivationLayer
from support.Functions import tanh, tanhDerivative, sigmoid, sigmoidDerivative, mse, \
mseDerivative, relu, reluDerivative, softmax, softmaxDerivative, crossEntropy, crossEntropyDerivative
from support.Bootstrap import bootstrap, standardize, transformY
from EvolutionaryAlgorithm import EvolutionaryAlgorithm
from NSGAII import NSGAII
from support.plotting_helper import plot_objectives, plot_LossIteration


# Data 
dataPath = '/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/Wholesale customers data.csv'

#def Wholesale_EA():
# load data
data = pd.read_csv(dataPath, sep= ',', index_col=False)
data = data.drop(["Region"],axis=1)

# standardize data
X = standardize(data.iloc[:,1:])
data_standardized = pd.concat([data.iloc[:,0], X],axis=1)

# bootstrap in train, vali and test
data_train, data_vali, data_test = bootstrap(data_standardized, replacement = False)

# transform y and X in np.array | y: 2=1, 1=0
#TRAIN
output = 2
y_train = transformY(data_train, output)
X_train = np.asanyarray(data_train.iloc[:,1:])

#VALIDATION
y_vali = transformY(data_vali, output)
X_vali = np.asanyarray(data_vali.iloc[:,1:])

#TEST
y_test = transformY(data_test, output)
X_test = np.asanyarray(data_test.iloc[:,1:])


# initialize 
popSize = 6 
it = 10 # iterations
minAcc = 0.90

#static variables in Class NeuralNetwork
NeuralNetwork.maxNeurons = 30
NeuralNetwork.minNeurons = 4
NeuralNetwork.maxHiddenLayers = 2
NeuralNetwork.sizeInput = 6
NeuralNetwork.sizeOutput = 2

EA = EvolutionaryAlgorithm(xTrain = X_train, yTrain = y_train, 
                       popSize = popSize)
NSGA = NSGAII()

initialPopulation = EA.randomPop(loss = crossEntropy, lossDerivative = crossEntropyDerivative) # random population

# main loop
population = initialPopulation

for t in range(it):
    print('** GENERATION',t+1,'**')
    # reproduction and mutation
    offSpring = EA.makeOffspring(population)
    
    # train with Adam
    EA.trainPop(population, epochs = 40, minAcc = minAcc)
    EA.trainPop(offSpring, epochs = 40, minAcc = minAcc) 
    minAcc += 0.02
   
    # evaluate on validation dataset 
    EA.predPop(offSpring, X = X_vali, Y = y_vali)
    EA.predPop(population, X = X_vali, Y = y_vali)
 
    # selection
    newPopParent = NSGA.run(population, offSpring)
    # get the test accuracy (only for plotting purpose)
    EA.predPop(newPopParent, X = X_test, Y = y_test, testSample = True)
    newPopParent.printPop()

    population = newPopParent
    
    
    
    
    

    
    
    
plot_objectives(population)
plt.show()

plot_LossIteration(population)
plt.show()



#if __name__ == "__main__":
#    print("Python version in use: ", sys.version)
#    print("Experiment_Wholesale")
#    print("##########-##########-##########")
#    Wholesale_EA()
#    print("##########-##########-##########")


