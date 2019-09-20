# Experiment.py

import pandas as pd
import numpy as np


from NeuralNetwork import NeuralNetwork
from support.Layer import Layer
from support.ActivationLayer import ActivationLayer
from support.Loss_n_ActivationFunction import tanh, tanhDerivative, sigmoid, sigmoidDerivative, mse, mseDerivative



inputLayer = Layer(6, 10)
activationFunction = ActivationLayer(sigmoid, sigmoidDerivative)
hiddenLayer = Layer(10, 30)
activationFunction2 = ActivationLayer(sigmoid, sigmoidDerivative)
hiddenLayer2 = Layer(30, 30)
activationFunction3 = ActivationLayer(sigmoid, sigmoidDerivative)
outputLayer = Layer(30, 3)
activationFunction4 = ActivationLayer(sigmoid, sigmoidDerivative)



layerList = [inputLayer, activationFunction, hiddenLayer, activationFunction2, hiddenLayer2, activationFunction3,
             outputLayer, activationFunction4]

nn= NeuralNetwork(layerList, mse, mseDerivative)

# Data

# random test input
Input_test = np.random.randn(2000,6)
Target_test = np.zeros((2000,3))+0.01
for i in range(len(Target_test)):
    n=np.random.randint(3)
    Target_test[i,n] = 0.99
    pass

df = pd.read_csv("/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/Wholesale customers data.csv", sep=',')


inputs = (np.asfarray(df.iloc[:,2:8]) / 112152.0 * 0.99) + 0.01 # scale and shift the inputs
targets = np.zeros((440,3)) + 0.01 # create target output values
for i in range(len(df)):
    targets[i,(df.iloc[i,1]-1)] = 0.99  # all_values[0] is the target label for this record
    pass

# Train

nn.train(inputs, targets, epochs = 10, batchSize = 60)

# test
#nn.predict(inputs[0:280])



