# Experiment.py

import pandas as pd
import numpy as np


from NeuralNetwork import NeuralNetwork
from support.Layer import Layer
from support.ActivationLayer import ActivationLayer
from support.Loss_n_ActivationFunction import tanh, tanhDerivative, sigmoid, sigmoidDerivative, mse, mseDerivative
from support.Bootstrap import bootstrap


inputLayer = Layer(3, 10)
activationFunction = ActivationLayer(tanh, tanhDerivative)
hiddenLayer = Layer(10, 5)
activationFunction2 = ActivationLayer(tanh, tanhDerivative)
#hiddenLayer2 = Layer(300, 30)
#activationFunction3 = ActivationLayer(sigmoid, sigmoidDerivative)
outputLayer = Layer(5, 2)
activationFunction4 = ActivationLayer(tanh, tanhDerivative)



layerList = [inputLayer, activationFunction, hiddenLayer, activationFunction2, 
             outputLayer, activationFunction4]

nn= NeuralNetwork(layerList, mse, mseDerivative)

## training data
#x_train = np.array([[0,0], [0,1], [1,0], [1,1]])
#y_train = np.array([[0], [1], [1], [0]])




#
inputs = np.array(([0.01,0.99,0.99],[0.99,0.99,0.01],[0.99,0.01,0.01],[0.01,0.01,0.99]))
#
#
targets = np.array(([0.99,0.01], [0.01,0.99], [0.01,0.99], [0.99,0.01]))


nn.train(inputs, targets, epochs = 100, Rprop = True)
#nn.predict(inputs, targets)

# load the data into a list
#df = pd.read_csv("/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/mnist_train_100.csv", sep=',', header=None, index_col=False)
#


# go through all records in the training data set



#
#
#inputs = (np.asfarray(df.iloc[:,1:]) / 255.0 * 0.99) + 0.01 # scale and shift the inputs 
#targets = np.zeros((100,10)) + 0.01 # create target output values
#for i in range(len(df)):
#    j = df.iloc[i,0]
#    j=int(j)
#    targets[i,j] = 0.99  # all_values[0] is the target label for this record
#    pass
#nn.learningRate = 0.35
#nn.train(inputs, targets, epochs = 10, Rprop = False)
#
#df_test = pd.read_csv("/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/mnist_test_10.csv", sep=',', header=None, index_col=False)
#inputs_test = (np.asfarray(df_test.iloc[:,1:]) / 255.0 * 0.99) + 0.01 # scale and shift the inputs 
#targets_test = np.zeros((10,10)) + 0.01 # create target output values
#for i in range(len(df_test)):
#    j = df_test.iloc[i,0]
#    j=int(j)
#    targets_test[i,j] = 0.99  # all_values[0] is the target label for this record
#    pass
#
#out = nn.predict(inputs, targets)

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
#
#for i in range(len(YHat)): 
#    print(i,":" , YHat[i] == Y[i])
#
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







# Data
#df = pd.read_csv("/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/zip.train.csv", sep=' ', header=None, index_col=False)
#
#df= df.iloc[:,0:257] 
#
#
## Train 
#
#inputs = df.iloc[:,1:257]
#inputs = inputs.as_matrix()
#
#
#targets = np.zeros((7291,10)) + 0.01 # create target output values
#for i in range(len(df)):
#    j = df.iloc[i,0]
#    j=int(j)
#    targets[i,j] = 0.99  # all_values[0] is the target label for this record
#    pass
#
#
#nn.train(inputs, targets, epochs = 10)
#
#
#


#
## random test input
#Input_test = np.random.randn(2000,6)
#Target_test = np.zeros((2000,3))+0.01
#for i in range(len(Target_test)):
#    n=np.random.randint(3)
#    Target_test[i,n] = 0.99
#    pass

#df = pd.read_csv("/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/Wholesale customers data.csv", sep=',')
#
#
#
#
#inputs = (np.asfarray(df.iloc[:,2:8]) / 112152.0 * 0.99) + 0.01 # scale and shift the inputs
#targets = np.zeros((440,3)) + 0.01 # create target output values
#for i in range(len(df)):
#    targets[i,(df.iloc[i,1]-1)] = 0.99  # all_values[0] is the target label for this record
#    pass
#
## Train
#
#nn.train(inputs, targets, epochs = 1, Rprop = True)
#
## test
#nn.predict(inputs[0:280], targets[0:280])



