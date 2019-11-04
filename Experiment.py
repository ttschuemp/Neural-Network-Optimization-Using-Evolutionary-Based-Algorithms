# Experiment.py

import pandas as pd
import numpy as np


from NeuralNetwork import NeuralNetwork
from support.Layer import Layer
from support.ActivationLayer import ActivationLayer
from support.Functions import tanh, tanhDerivative, sigmoid, sigmoidDerivative, mse, \
mseDerivative, relu, reluDerivative, softmax, softmaxDerivative, crossEntropy, crossEntropyDerivative
from support.Bootstrap import bootstrap
import matplotlib
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

# Data 
dataPath = '/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/mnist_train (1).csv'

df = pd.read_csv(dataPath, sep=',', header=None, index_col=False)
df = df.iloc[0:2500,:]


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

##

# initialize NN

inputLayer = Layer(784, 100)
activationFunction = ActivationLayer(tanh, tanhDerivative)
hiddenLayer = Layer(100, 20)
activationFunction2 = ActivationLayer(tanh, tanhDerivative)
hiddenLayer2 = Layer(20, 10)
activationFunction3 = ActivationLayer(tanh, tanhDerivative)
outputLayer = Layer(10, 10)
activationFunction4 = ActivationLayer(softmax, softmaxDerivative)



layerList = [inputLayer, activationFunction, hiddenLayer, activationFunction2, hiddenLayer2, activationFunction3, outputLayer, activationFunction4]

nn= NeuralNetwork(layerList, crossEntropy, crossEntropyDerivative)


nn.train(X, Y, epochs = 9)


pred_vali = nn.predict(X_vali, Y_vali)
pred_vali_max = np.argmax(pred_vali, axis = 1)



#nnQp.train(inputs, targets, epochs = 2, learningAlgorithm = "Quickpro")
#
#nnRP.train(inputs, targets, epochs = 2, learningAlgorithm = "Rprop")
#yhat_train=nn.predict(xTrain, yTrain)
#yhat_train=np.argmax(yhat_train, axis =1)
#y = np.argmax(yTrain, axis =1)
#
#accuracyIS = classification(y, yhat_train)
#
#yhat_test=nn.predict(xTest, yTest)
#yhat_test=np.argmax(yhat_test, axis =1)
#y_test = np.argmax(yTest, axis =1)
#
#accuracyOOS = classification(y_test, yhat_test)


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



