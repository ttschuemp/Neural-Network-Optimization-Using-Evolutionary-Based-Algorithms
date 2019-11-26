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



#def artificialData_Exp():
# load data
X_train = np.load('/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/train_inputs.npy').T
y_train_ = np.load('/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/train_targets.npy').T
y_train = transformY_ad(y_train_,2)

X_test = np.load('/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/test_inputs.npy').T
y_test_ = np.load('/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/test_targets.npy').T
y_test = transformY_ad(y_test_,2)

#    plt.scatter(X_train[:,0],X_train[:,1], c=np.argmax(y_train,axis = 1))
#    plt.scatter(X_test[:,0],X_test[:,1], marker='>', c=np.argmax(y_test,axis = 1))
#    ax = plt.gca()
#    ax.set_facecolor('xkcd:salmon')
#    ax.set_facecolor((1.0, 0.47, 0.42))
#    plt.show()

# define neural network
inputLayer = Layer(2, 8)
activationFunction = ActivationLayer(relu, reluDerivative)
hiddenLayer = Layer(8, 8)
activationFunction2 = ActivationLayer(relu, reluDerivative)
outputLayer = Layer(8, 2)
activationFunction4 = ActivationLayer(softmax, softmaxDerivative)

layerList = [inputLayer, activationFunction, hiddenLayer, activationFunction2, outputLayer, activationFunction4]
nn = NeuralNetwork(layerList, crossEntropy, crossEntropyDerivative)

# train, test
nn.train(X_train, y_train, epochs = 600) 
pred = nn.predict(X_test, y_test, testSample = True)

# simulate data with these trained parameters (true DGP)
d = 2
n = 1500
artificialData_X = (np.random.rand(n, d)-0.5)*2
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
minAcc = 0.9 # makes algorithm much faster!! and more fair comparison! Better convergence

for e in range(10): # 10 experiments
    
    NSGA = NSGAII()
    EA = EvolutionaryAlgorithm(xTrain = X_train, yTrain = y_train, 
                           popSize = popSize)
    initialPopulation = EA.randomPop(loss = crossEntropy, lossDerivative = crossEntropyDerivative) # random population
    # main loop
    population = initialPopulation
    
    for t in range(it):
        print('** GENERATION',t+1,'**')
        # reproduction and mutation
        offSpring = EA.makeOffspring(population, t-4)
        
        # train with Adam
        EA.trainPop(population, epochs = 2, minAcc = minAcc) 
        EA.trainPop(offSpring, epochs = 2, minAcc = minAcc) 
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

plot_AD(Populations, it)
plt.show()




#if __name__ == "__main__":
#    print("##########-##########-##########")
#    print("Artificial Data Experiment")
#    artificialData_Exp()
#    print("##########-##########-##########")
#          
#          
          
          
          
#


#weights = Populations[1].neuralNetworks[0].layers[4].weights
#with open('weights1.csv', 'w') as csvFile:
#    writer = csv.writer(csvFile)
#    writer.writerows(weights)
#csvFile.close()
#dataPath = '/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/flower.mat'
#
#def Flower_Exp():
#    
#    # load data
#    train = (scio.loadmat(dataPath)['train']).T
#    n, d = train.shape
#    y_train = train[:,-1].reshape(n,1)
#    y_train = transformY_flower(y_train,2)
#    X_train = train[:,0:-1]
#
#    test = (scio.loadmat(dataPath)['test']).T
#    n, d = test.shape
#    y_test = test[:,-1].reshape(n,1)
#    y_test = transformY_flower(y_test,2)
#    X_test = test[:,0:-1]
#
#    # define neural network
#    inputLayer = Layer(2,15)
#    activationFunction = ActivationLayer(tanh, tanhDerivative)
#    hiddenLayer = Layer(15, 15)
#    activationFunction2 = ActivationLayer(tanh, tanhDerivative)
##    hiddenLayer2 = Layer(10, 10)
##    activationFunction3 = ActivationLayer(tanh, tanhDerivative)
#    outputLayer = Layer(15, 2)
#    activationFunction4 = ActivationLayer(softmax, softmaxDerivative)
#
#    layerList = [inputLayer, activationFunction, hiddenLayer, activationFunction2, outputLayer, activationFunction4]
#    nn = NeuralNetwork(layerList, crossEntropy, crossEntropyDerivative)
#    
#    # train, test
#    nn.train(X_train, y_train, epochs = 200)
#    pred = nn.predict(X_test, y_test, testSample = True)
#
#    # plot
##    grid_xlim = [np.min(X_train[:, 0]), np.max(X_train[:, 0])]
##    grid_ylim = [np.min(X_train[:, 1]), np.max(X_train[:, 1])]
##    plt.scatter(X_train[:,0], X_train[:,1], c = np.argmax(y_train,axis = 1))
##    plot_decision_function(X_train = X_train, y_train = y_train,
##                           grid_xlim = grid_xlim, grid_ylim = grid_ylim, save_path=None)
#    
