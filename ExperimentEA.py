# ExperimentEA.py

import numpy as np
import pandas as pd
from EvolutionaryAlgorithm import EvolutionaryAlgorithm




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
    pass


EA = EvolutionaryAlgorithm(epochs = 3, xTrain = inputs, yTrain = targets, 
                           popSize = 20, xTest = inputs_test, yTest =targets_test)
iterations = 200


initialPopulation = EA.randomPop()


for i in range(iterations):

    EA.trainPop(initialPopulation)
    
    offSpring = EA.makeOffspring(initialPopulation)
    EA.predPop(offSpring)
    EA.predPop(initialPopulation)
    
    newPopParent=EA.updatePop(initialPopulation, offSpring)
    
    initalPopulation= newPopParent
    
    print(newPopParent.evaluatePop())
    pass
