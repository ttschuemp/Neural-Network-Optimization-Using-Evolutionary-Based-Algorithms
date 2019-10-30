# ExperimentEA2.py
import matplotlib
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import numpy as np
import pandas as pd
from EvolutionaryAlgorithm import EvolutionaryAlgorithm
from NSGAII import NSGAII
from support.Bootstrap import bootstrap



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
EA = EvolutionaryAlgorithm(epochs = 6, xTrain = xTrain, yTrain = yTrain, 
                           popSize = 10, xTest = xTest, yTest = yTest)


colours = ['bo', 'gx', 'r*', 'cv', 'm1', 'y2', 'k3', 'w4']

# random initial population
initialPopulation = EA.randomPop(noHiddenLayers = True)
nsga = NSGAII()

#initialPopcopy = initialPopulation.copy_pop(initialPopulation)
population = initialPopulation


## search ##
iterations = 10
for i in range(iterations):
  
    # train population
    EA.trainPop(population, learningAlgorithm = "BP")
#    popTrained_copy = population.copy_pop(population)
    # reproduction and mutation
    offSpring = EA.makeOffspring(population)
#    OSpop_copy =offSpring.copy_pop(offSpring)
    # train off spring
    EA.trainPop(offSpring, learningAlgorithm = "BP") 
    EA.trainPop(offSpring, learningAlgorithm = "BP")
#    OSpopTrained = offSpring.copy_pop(offSpring)
# 
    # evaluation & selection
#    newPopParent = EA.updatePop(population, offSpring)
    newPopParent = nsga.run(population, offSpring)
#    
    if newPopParent == "front to small": 
        continue
#    
#    newParentNSGA = newPopParent.copy_pop(newPopParent)
    population = newPopParent
#    for n in population.neuralNetworks:
#        plt.plot(n.err,n.nrNeurons, colours[i], markersize=10)


## report ##
    print("generation: ",i+1 , population.evaluatePop())
#plt.axis([0,1 , 0,2000])
#plt.show()
EA.predPop(population)
population.evaluatePop()
# fig1, axes = plt.subplots(1,2 figsize=(10,5))
# axes[0].scatter(year, price)
# plt.show()
