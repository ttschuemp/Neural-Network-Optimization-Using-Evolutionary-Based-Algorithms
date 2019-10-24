# ExperimentEA2.py

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

df = pd.read_csv("/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/mnist_train (1).csv", sep=',', header=None, index_col=False)
dfTrain, dfTest = bootstrap(df.iloc[0:1500,:])


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


## initialize ##
EA = EvolutionaryAlgorithm(epochs = 5, xTrain = xTrain, yTrain = yTrain, 
                           popSize = 20, xTest = xTest, yTest = yTest)


colours = ['bo', 'gx', 'r*', 'cv', 'm1', 'y2', 'k3', 'w4']

# random initial population
initialPopulation = EA.randomPop()
nsga = NSGAII()

population = initialPopulation


## search ##
iterations = 15
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
