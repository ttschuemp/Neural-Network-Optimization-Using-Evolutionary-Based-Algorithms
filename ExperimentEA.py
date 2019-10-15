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

df = pd.read_csv("/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/Wholesale customers data.csv", sep=',')


inputs = (np.asfarray(df.iloc[:,2:8]) / 112152.0 * 0.99) + 0.01 # scale and shift the inputs
targets = np.zeros((440,3)) + 0.01 # create target output values
for i in range(len(df)):
    targets[i,(df.iloc[i,1]-1)] = 0.99  # all_values[0] is the target label for this record
    pass

dfTrain, dfTest = bootstrap(df)

inputsTrain = dfTrain.iloc[:,2:]
targetsTrain = dfTrain[['Region']]

inputsTest = dfTest.iloc[:,2:]
targetsTest = dfTest[['Region']]

## initialize ##
EA = EvolutionaryAlgorithm(epochs = 2, xTrain = inputsTrain, yTrain = targetsTrain, 
                           popSize = 30, xTest = inputsTest, yTest = targetsTest)


colours = ['bo', 'gx', 'r*', 'cv', 'm1', 'y2', 'k3', 'w4']

# random initial population
initialPopulation = EA.randomPop()
nsga = NSGAII()

population = initialPopulation


## search ##
iterations = 3
for i in range(iterations):
  
    # train population
    EA.trainPop(population)
 
    # reproduction and mutation
    offSpring = EA.makeOffspring(population)
  
    # train off spring
    EA.trainPop(offSpring) # RuntimeWarning: overflow
   
    # predict on test data set
    EA.predPop(offSpring)
    EA.predPop(population)
 
    # evaluation & selection
#    newPopParent = EA.updatePop(population, offSpring)
    newPopParent = nsga.run(population, offSpring)
 
    population = newPopParent
    for n in population.neuralNetworks:
        plt.plot(n.err,n.nrNeurons, colours[i], markersize=10)


## report ##
    print("generation: ",i+1 , population.evaluatePop())
plt.axis([0,1 , 0,2000])
plt.show()

# fig1, axes = plt.subplots(1,2 figsize=(10,5))
# axes[0].scatter(year, price)
# plt.show()
