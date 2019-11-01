#Experiment_NeuralNetwork.py
import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import scipy.io as scio
import math
from NeuralNetwork import NeuralNetwork
from support.Layer import Layer
from support.ActivationLayer import ActivationLayer
from support.Loss_n_ActivationFunction import tanh, tanhDerivative, sigmoid, sigmoidDerivative, mse, mseDerivative

from matplotlib.patches import Ellipse


plt.style.use("seaborn-whitegrid")


dataPath = '/Users/tobiastschuemperlin/Documents/Master WWZ/Masterarbeit/Python/Datasets/'

#------------------------------------------------------------------------------
def classification(y, estimation):
    y=y[:,None]
    b= y == estimation
    
    return b.sum()/len(estimation)


def print_classification(y, estimation):
    # Helper function to check if data items are classified correctly
    correctly_classified = 0
    false_class_0 = 0
    false_class_1 = 0
    accuracyIS = 0

    _, n = np.shape(y)

    for i in range(n):
        if classification(y, estimation):
            if labels[i] == 1:
                correctly_classified += 1
            else:
                false_class_1 += 1

        else:
            if labels[i] == 0:
                correctly_classified += 1
            else:
                false_class_0 += 1

    print('############# DATA CLASSIFICATION #################')
    print('Classified', correctly_classified, 'out of', n, 'samples correctly.')
    print('False omega_1:', false_class_0)
    print('False omega_2:', false_class_1)
    print('####################################################')

#------------------------------------------------------------------------------
# Run the 'main':
if __name__ == '__main__':
    # Load of the trainings/test-set:
    data_train = scio.loadmat(os.path.join(dataPath, 'mle_toy_train.mat'))
    data_test = scio.loadmat(os.path.join(dataPath, 'mle_toy_test.mat'))

    # Split up the data of the mle_toy_train:
    class0_training_data = data_train['omega_1']
    class0_training_labels = np.zeros((1, np.shape(class0_training_data)[1]))
    class1_training_data = data_train['omega_2']
    class1_training_labels = np.ones((1, np.shape(class1_training_data)[1]))
    all_training_data = np.append(class0_training_data, class1_training_data, axis=1)
    all_training_labels = np.append(class0_training_labels, class1_training_labels)

    class0_test_data = data_test['omega_1']
    class0_test_labels = np.zeros((1, np.shape(class0_test_data)[1]))
    class1_test_data = data_test['omega_2']
    class1_test_labels = np.ones((1, np.shape(class1_test_data)[1]))
    all_test_data = np.append(class0_test_data, class1_test_data, axis=1)
    all_test_labels = np.append(class0_test_labels, class1_test_labels)
    
    
    # initialize NN

    # (inputSize,output)
    layerList = [Layer(2, 5), ActivationLayer(tanh, tanhDerivative), 
                 Layer(5, 1), ActivationLayer(sigmoid, sigmoidDerivative)]
               #(input, outputSize)
               
    
    nn= NeuralNetwork(layerList)
    
    
    # Train NN
    
    nn.train(all_training_data.T, all_training_labels, epochs= 30)
    yEst = nn.predict(all_training_data.T, all_training_labels)
    yEst_rounded = np.round(yEst)
    
    accuracyIS=classification(all_training_labels, yEst_rounded)





    # visualize the data set
#    plt.clf() # clears the figure befor plotting it again
    fig1 = plt.figure(1)
    plt.plot(class0_training_data[0, :], class0_training_data[1, :], 'bx')
    plt.plot(class1_training_data[0, :], class1_training_data[1, :], 'r.')
    plt.legend(['$\omega_1$ Training', '$\omega_2$ Training'])
    plt.title('Training Data')
    plt.show(block=False)
    
#    print("\nTraining data classification:")
#    print_classification(all_training_data, all_training_labels, class0_mean, class0_cov, class1_mean, class1_cov)
#
#    print("\nTest data classification:")
#    print_classification(all_test_data, all_test_labels, class0_mean, class0_cov, class1_mean, class1_cov)
#
#
#    print("\nClass0 Sample: {}, pdf-class0: {}, pdf-class1: {}".format(
#        class0_random_sample,
#        pdf(class0_random_sample, class0_mean, class0_cov),
#        pdf(class0_random_sample, class1_mean, class1_cov))
#    )
#
#    print("\nClass1 Sample: {}, pdf-class0: {}, pdf-class1: {}".format(
#        class1_random_sample,
#        pdf(class1_random_sample, class0_mean, class0_cov),
#        pdf(class1_random_sample, class1_mean, class1_cov))
#    )
#
#    # visualize estimated normal distributions
#    fig2 = plt.figure(2)
#    ax = plt.axes()
#    plt.plot(class0_training_data[0, :], class0_training_data[1, :], 'bx')
#    plt.plot(class0_test_data[0, :], class0_test_data[1, :], 'bo')
#    plt.plot(class0_random_sample[0], class0_random_sample[1], 'k*')
#    plot_cov(ax, class0_mean, class0_cov, 'blue')
#
#    plt.plot(class1_training_data[0, :], class1_training_data[1, :], 'rx')
#    plt.plot(class1_test_data[0, :], class1_test_data[1, :], 'ro')
#    plt.plot(class1_random_sample[0], class1_random_sample[1], 'gD')
#    plot_cov(ax, class1_mean, class1_cov, 'red')
#    plt.legend(['$\omega_0$ Training', '$\omega_0$ Test', '$\omega_0$ Sample',
#                '$\omega_1$ Training', '$\omega_1$ Test', '$\omega_1$ Sample'])
#    plt.title('Training and test Data')
#    plt.show()
