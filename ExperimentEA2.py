# ExperimentEA2.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

from NeuralNetwork import NeuralNetwork
from support.Layer import Layer
from support.ActivationLayer import ActivationLayer
from support.Functions import tanh, tanhDerivative, sigmoid, sigmoidDerivative, mse, \
mseDerivative, relu, reluDerivative, softmax, softmaxDerivative, crossEntropy, crossEntropyDerivative
from support.Bootstrap import bootstrap, standardize, transformY_mnist
from EvolutionaryAlgorithm import EvolutionaryAlgorithm
from NSGAII import NSGAII
from support.plotting_helper import plot_objectives, plot_IterationSGD, plot_testAcc, plot_exploration



#--------------------------------------------------------------------------------

df = pd.read_csv("/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/mnist_train (1).csv", sep=',', header=None, index_col=False)
dfTrain, dfTest, dfVali = bootstrap(df.iloc[0:1500,:])


inputs = (np.asfarray(dfTrain.iloc[:,1:]) / 255.0 * 0.99) + 0.01 # scale and shift the inputs 
targets = np.zeros((len(dfTrain),10)) + 0.01 # create target output values
for i in range(len(dfTrain)):
    j = dfTrain.iloc[i,0]
    j=int(j)
    targets[i,j] = 0.99  # all_values[0] is the target label for this record
    pass



xTrain = inputs[0:1000,:]
xTest = inputs[1000:1100,:]
yTrain = targets[0:1000,:]
yTest = targets[1000:1100,:]

Populations = []
EAs = []
## initialize ##
EA = EvolutionaryAlgorithm(xTrain = xTrain, yTrain = yTrain, 
                           popSize = 5)
NeuralNetwork.maxNeurons = 400 
NeuralNetwork.minNeurons = 30 
NeuralNetwork.maxHiddenLayers = 4
NeuralNetwork.sizeInput = 784
NeuralNetwork.sizeOutput = 10

colours = ['bo', 'gx', 'r*', 'cv', 'm1', 'y2', 'k3', 'w4']

# random initial population
NSGA = NSGAII()

initialPopulation = EA.randomPop(loss = crossEntropy, lossDerivative = crossEntropyDerivative) # random population
# main loop
population = initialPopulation
it = 5
minAcc = 0.9

for t in range(it):
    print('** GENERATION',t+1,'**')
    # reproduction and mutation

    # reproduction and mutation
    offSpring = EA.makeOffspring(population)

    # train with Adam
    EA.trainPop(population, epochs = 15, minAcc = minAcc) 
    EA.trainPop(offSpring, epochs = 15, minAcc = minAcc) 
    minAcc += 0.02 
    
    # evaluate on validation dataset 
    EA.predPop(offSpring, X = xTest, Y = yTest)
    EA.predPop(population, X = xTest, Y = yTest)
 
    # selection
    newPopParent = NSGA.run(population, offSpring)
    # get the test accuracy (only for plotting)
    EA.predPop(newPopParent, X = xTest, Y = yTest, testSample = True)
    newPopParent.printPop()

    population = newPopParent
    
Populations.append(population)
EAs.append(EA)
    
#plot_swarm(population)
#plt.show()
    
plot_objectives(Populations)
plt.show()
    
plot_IterationSGD(Populations)
plt.show()
    
plot_testAcc(Populations)
plt.show()
        
plot_exploration(EAs, it)
plt.show()

