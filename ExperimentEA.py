# ExperimentEA.py
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from EvolutionaryAlgorithm import EvolutionaryAlgorithm
from NSGAII import NSGAII
from support.Bootstrap import bootstrap


matplotlib.use('MacOSX')
plt.style.use("seaborn-whitegrid")
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


#--------------------------------------------------------------------------------
inputs = np.array(([0.01,0.99],[0.99,0.01],[0.99,0.99],[0.01,0.01]))
targets = np.array(([0.99], [0.99], [0.01], [0.01]))

xTrain = inputs
xTest = inputs
yTrain = targets
yTest = targets


## initialize ##
EA = EvolutionaryAlgorithm(epochs = 1, xTrain = xTrain, yTrain = yTrain, 
                           popSize = 10, xTest = xTest, yTest = yTest)


colours = ['bo', 'gx', 'r*', 'cv', 'm1', 'y2', 'k3', 'w4']

# random initial population
initialPopulation = EA.randomPop(noHiddenLayers = True)
nsga = NSGAII()

population = initialPopulation


## search ##
iterations = 20
for i in range(iterations):
  
    # train population
    EA.trainPop(population, learningAlgorithm = "BP")
 
    # reproduction and mutation
    offSpring = EA.makeOffspring(population)
  
    # train off spring
    EA.trainPop(offSpring, learningAlgorithm = "BP") # RuntimeWarning: overflow
   
    # predict on test data set
#    EA.predPop(offSpring)
#    EA.predPop(population)
 
    # evaluation & selection
#    newPopParent = EA.updatePop(population, offSpring)
    newPopParent = nsga.run(population, offSpring)
#    
    if newPopParent == "front to small": 
        continue
#    
    population = newPopParent
#    for n in population.neuralNetworks:
#        plt.plot(n.err,n.nrNeurons, colours[i], markersize=10)


## report ##
    print("generation: ",i+1 , population.evaluatePop())
#plt.axis([0,1 , 0,2000])
#plt.show()

# fig1, axes = plt.subplots(1,2 figsize=(10,5))
# axes[0].scatter(year, price)
# plt.show()
