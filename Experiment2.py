# Experiment2.py

import pandas as pd
import numpy as np


from NeuralNetwork import NeuralNetwork
from support.Layer import Layer
from support.ActivationLayer import ActivationLayer
from support.Loss_n_ActivationFunction import tanh, tanhDerivative, sigmoid, sigmoidDerivative, mse, mseDerivative
from support.Bootstrap import bootstrap


inputLayer = Layer(784, 100)
activationFunction = ActivationLayer(tanh, tanhDerivative)
hiddenLayer = Layer(100, 180)
activationFunction2 = ActivationLayer(tanh, tanhDerivative)
hiddenLayer2 = Layer(180, 80)
activationFunction3 = ActivationLayer(tanh, tanhDerivative)
outputLayer = Layer(80, 10)
activationFunction4 = ActivationLayer(tanh, tanhDerivative)



layerList = [inputLayer, activationFunction, hiddenLayer, activationFunction2, hiddenLayer2, activationFunction3, 
             outputLayer, activationFunction4]

n2= NeuralNetwork(layerList, mse, mseDerivative)

nRprop= NeuralNetwork(layerList, mse, mseDerivative)
n= NeuralNetwork(layerList, mse, mseDerivative)


df = pd.read_csv("/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/mnist_train (1).csv", sep=',', header=None, index_col=False)
dfTrain, dfTest = bootstrap(df.iloc[0:1500,:])


inputs = (np.asfarray(dfTrain.iloc[:,1:]) / 255.0 * 0.99) + 0.01 # scale and shift the inputs 
targets = np.zeros((len(dfTrain),10)) + 0.01 # create target output values
for i in range(len(dfTrain)):
    j = dfTrain.iloc[i,0]
    j=int(j)
    targets[i,j] = 0.99  # all_values[0] is the target label for this record
    pass


#n2.learningRate = 0.25
#
#n2.train(inputs[0:1000,:], targets[0:1000,:], epochs = 2, learningAlgorithm = "BP")
#
#nRprop.train(inputs[0:1000,:], targets[0:1000,:], epochs = 2, learningAlgorithm = "Rprop")
n.learningRate = 0.45
n.train(inputs[0:1000,:], targets[0:1000,:], epochs = 1, learningAlgorithm = "BP")

#df_test = pd.read_csv("/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/mnist_test (1).csv", sep=',', header=None, index_col=False)
#inputs_test = (np.asfarray(df_test.iloc[:,1:]) / 255.0 * 0.99) + 0.01 # scale and shift the inputs 
#targets_test = np.zeros((len(df_test),10)) + 0.01 # create target output values
#for i in range(len(df_test)):
#    j = df_test.iloc[i,0]
#    j=int(j)
#    targets_test[i,j] = 0.99  # all_values[0] is the target label for this record
#    pass

#n2.predict(inputs[1000:1100,:], targets[1000:1100,:])


#
## Prediction Test
#YHat= []
#for i in range(len(out)):
#    yHat = np.argmax(out[i])
#    YHat.append(yHat)
#YHat = np.array(YHat)
#
#
#
## Target Test
#Y= []
#for i in range(len(targets_test)):
#    y = np.argmax(targets_test[i])
#    Y.append(y)
#Y = np.array(Y)
#
#scorecard = []
#for i in range(len(YHat)): 
#    if (YHat[i] == Y[i]): 
#        scorecard.append(1)
#    else:                         
#        scorecard.append(0)
#        pass
#
#scorecard_array = np.asarray(scorecard)
#print ("performance OoS = ", scorecard_array.sum() /scorecard_array.size)
#
#
#
#
#

