# ExperimentEA.py

#import matplotlib
#matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import pandas as pd
from EvolutionaryAlgorithm import EvolutionaryAlgorithm
import numpy as np 
from NeuralNetwork import NeuralNetwork
from support.Layer import Layer
from support.ActivationLayer import ActivationLayer
from support.Functions import tanh, tanhDerivative, sigmoid, sigmoidDerivative, mse, \
mseDerivative, relu, reluDerivative, softmax, softmaxDerivative, crossEntropy, crossEntropyDerivative
from support.Bootstrap import bootstrap, standardize, transformY_mnist
from EvolutionaryAlgorithm import EvolutionaryAlgorithm
from NSGAII import NSGAII
from support.plotting_helper import plot_objectives, plot_IterationSGD, plot_testAcc, plot_exploration


#----------------------------------------------------------------------------------
# Data 
dataPath = '/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/mnist_train (1).csv'

df = pd.read_csv(dataPath, sep=',', header=None, index_col=False)
df = df.iloc[0:500,:]


inputs = (np.asfarray(df.iloc[:,1:]) / 255.0 * 0.99) + 0.01 # scale and shift the inputs 
targets = np.zeros((len(df),10)) + 0.01 # create target output values
for i in range(len(df)):
    j = df.iloc[i,0]
    j=int(j)
    targets[i,j] = 0.99  # all_values[0] is the target label for this record
    pass



dfnew=np.concatenate((inputs,targets), axis=1)
pdDfNew = pd.DataFrame(dfnew)
dfTrain, dfvalidation, dfTest = bootstrap(pdDfNew)
del inputs, targets, i, j, pdDfNew, dfnew


dfTrain_np= np.asarray(dfTrain)
dfvalidation_np = np.asarray(dfvalidation) 
dfTest_np = np.asarray(dfTest) 

X = dfTrain_np[:,0:784]
Y = dfTrain_np[:,784:]

X_vali = dfvalidation_np[:,0:784]
Y_vali = dfvalidation_np[:,784:]

NeuralNetwork.maxNeurons = 300 
NeuralNetwork.minNeurons = 50 
NeuralNetwork.maxHiddenLayers = 4
NeuralNetwork.sizeInput = 784
NeuralNetwork.sizeOutput = 10


## initialize ##
EA = EvolutionaryAlgorithm(xTrain = X, yTrain = Y, 
                           popSize = 10)


colours = ['bo', 'gx', 'r*', 'cv', 'm1', 'y2', 'k3', 'w4']

# random initial population
initialPopulation = EA.randomPop(loss = crossEntropy, lossDerivative = crossEntropyDerivative)
nsga = NSGAII()

population = initialPopulation


## search ##
it = 5
z = 0.90
for i in range(it):
  
    # train population
#    EA.trainPop(population, epochs = 6) # train until acc. training set min 90 %
 
    # reproduction and mutation
    offSpring = EA.makeOffspring(population)
    
    # train off spring
    EA.trainPop(population, epochs = 15, minAcc = z)
    EA.trainPop(offSpring, epochs = 15, minAcc = z) 
    z = z + 0.02
   
    # predict on validation data set (dont train the weights on this dataset only for hyperparameter adjustment)
    EA.predPop(offSpring, X = X_vali, Y = Y_vali)
    EA.predPop(population, X = X_vali, Y = Y_vali)
 
    # evaluation & selection
#    newPopParent = EA.updatePop(population, offSpring)
    newPopParent = nsga.run(population, offSpring)
    newPopParent.printPop()
#    
    population = newPopParent
    for n in population.neuralNetworks:
        plt.plot(n.accuracyOOS,n.nrNeurons, colours[i], markersize=5)
    

### report ##
#    print("generation: ",i+1 , population.evaluatePop())
#plt.axis([0,1 , 0,2000])

plt.show()
# fig1, axes = plt.subplots(1,2 figsize=(10,5))
# axes[0].scatter(year, price)
# plt.show()
    

#for n in offSpring.neuralNetworks:
#    print(n.accuracyTrain)

# TEST 




#matplotlib.use('MacOSX')
#plt.style.use("seaborn-whitegrid")
#--------------------------------------------------------------------------------
#df = pd.read_csv("/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/mnist_train_100.csv", sep=',', header=None, index_col=False)
#
#
#inputs = (np.asfarray(df.iloc[:,1:]) / 255.0 * 0.99) + 0.01 # scale and shift the inputs 
#targets = np.zeros((100,10)) + 0.01 # create target output values
#for i in range(len(df)):
#    j = df.iloc[i,0]
#    j=int(j)
#    targets[i,j] = 0.99  # all_values[0] is the target label for this record
#    pass
#
#
#df_test = pd.read_csv("/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/mnist_test_10.csv", sep=',', header=None, index_col=False)
#inputs_test = (np.asfarray(df_test.iloc[:,1:]) / 255.0 * 0.99) + 0.01 # scale and shift the inputs 
#targets_test = np.zeros((10,10)) + 0.01 # create target output values
#for i in range(len(df_test)):
#    j = df_test.iloc[i,0]
#    j=int(j)
#    targets_test[i,j] = 0.99  # all_values[0] is the target label for this record
#    
#--------------------------------------------------------------------------------
#df = pd.read_csv("/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/Wholesale customers data.csv", sep=',')
#
#
#dfTrain, dfTest = bootstrap(df)
#
#xTrainDf= dfTrain.iloc[:,2:]
#yTrainDf = dfTrain[['Region']]
#
#xTestDf = dfTest.iloc[:,2:]
#yTestDf = dfTest[['Region']]
#
## Transform input 
#xTrainDf = (xTrainDf/112152.0*0.99) + 0.01 # scale and shift the inputs
#
#r,c = yTrainDf.shape
#index= np.asarray(yTrainDf)
#yTrainDf = np.zeros((r,3)) + 0.01 # create target output values
#for i in range(r):
#    yTrainDf[i,(index[i]-1)] = 0.99  # all_values[0] is the target label for this record
#    pass
#
#xTestDf = (xTestDf/112152.0*0.99) + 0.01 # scale and shift the inputs
#
#r,c = yTestDf.shape
#index= np.asarray(yTestDf)
#yTestDf = np.zeros((r,3)) + 0.01 # create target output values
#for i in range(r):
#    yTestDf[i,(index[i]-1)] = 0.99  # all_values[0] is the target label for this record
#    pass
#
#inputsTrain = np.asfarray(xTrainDf)
#inputsTest = np.asfarray(xTestDf)
#targetsTrain = yTrainDf
#targetsTest = yTestDf


##--------------------------------------------------------------------------------
#inputs = np.array(([0.01,0.99],[0.99,0.01],[0.99,0.99],[0.01,0.01]))
#targets = np.array(([0.99], [0.99], [0.01], [0.01]))
#
#xTrain = inputs
#xTest = inputs
#yTrain = targets
#yTest = targets


