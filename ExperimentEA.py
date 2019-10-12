# ExperimentEA.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from EvolutionaryAlgorithm import EvolutionaryAlgorithm
from Population import Population
from NSGAII import NSGAII




df = pd.read_csv("/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/mnist_train_100.csv", sep=',', header=None, index_col=False)


inputs = (np.asfarray(df.iloc[:,1:]) / 255.0 * 0.99) + 0.01 # scale and shift the inputs 
targets = np.zeros((100,10)) + 0.01 # create target output values
for i in range(len(df)):
    j = df.iloc[i,0]
    j=int(j)
    targets[i,j] = 0.99  # all_values[0] is the target label for this record
    pass


df_test = pd.read_csv("/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/mnist_test_10.csv", sep=',', header=None, index_col=False)
inputs_test = (np.asfarray(df_test.iloc[:,1:]) / 255.0 * 0.99) + 0.01 # scale and shift the inputs 
targets_test = np.zeros((10,10)) + 0.01 # create target output values
for i in range(len(df_test)):
    j = df_test.iloc[i,0]
    j=int(j)
    targets_test[i,j] = 0.99  # all_values[0] is the target label for this record
    


## initialize ##
EA = EvolutionaryAlgorithm(epochs = 3, xTrain = inputs, yTrain = targets, 
                           popSize = 10, xTest = inputs_test, yTest =targets_test)
iterations = 20

# random initial population
initialPopulation = EA.randomPop()
nsga = NSGAII()

population = initialPopulation
## search ##
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
## report ##
    print("generation: ",i , population.evaluatePop())
    


