#plotting_helper.py

import numpy as np
import matplotlib.pyplot as plt

def plot_objectives(population): 
    for n in population.neuralNetworks:
        plt.plot(n.nrNeurons, n.accuracyTrain)
