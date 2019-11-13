#plotting_helper.py

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

z = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
#z = [1 ,0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

def plot_objectives(population): 
    leng= len(population.neuralNetworks[0].accuracyTrain_h)
    fig1, axes = plt.subplots(1,2, figsize=(10,5))
    fig1.suptitle('Validation and Test')
    for i in range(leng):
        for n in population.neuralNetworks:
            u = np.random.rand(4)*0.05
            axes[0].scatter(n.accuracyVali_h[i]+u[0], n.nrNeurons_h[i]+u[1], alpha = z[i], color = 'midnightblue')
            axes[1].scatter(n.accuracyTest_h[i]+u[3], n.nrNeurons_h[i]+u[2], alpha = z[i],color = 'midnightblue')
            axes[0].set_ylabel('Number of Neurons')
            axes[0].set_xlabel('Accuracy Validation Data')
            axes[1].set_ylabel('Number of Neurons')
            axes[1].set_xlabel('Accuracy Test Data')


def plot_LossIteration(population): 
    fig2, axes = plt.subplots(1,2, figsize=(10,5))
    for n in population.neuralNetworks:
        it = np.array(range(len(n.err_h)))
        axes[0].plot(it, n.err_h, color = 'midnightblue', alpha = 0.2)
        axes[1].plot(it, n.accuracyTrain_h_iteration, color = 'midnightblue', alpha = 0.2)
        axes[0].set_ylabel('Error')
        axes[0].set_xlabel('Steps')
        axes[1].set_ylabel('Accuracy Trainin Data')
        axes[1].set_xlabel('Steps')
# still need to add the average over the top 5 
# different population size 


