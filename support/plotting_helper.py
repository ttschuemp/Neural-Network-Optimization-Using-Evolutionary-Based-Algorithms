#plotting_helper.py

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

z = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
#z = [1 ,0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

colors = ['midnightblue', 'darkblue', 'darkslateblue', 'slateblue', 'mediumorchid', 'purple', 'plum', 'orchid', 'pink', 'lavenderblush']


colours =['#393b79','#5254a3', '#6b6ecf', '#9c9ede', '#637939', '#8ca252', '#b5cf6b', '#cedb9c', '#8c6d31', '#bd9e39', '#e7ba52','#e7cb94', '#843c39', '#ad494a', '#d6616b','#e7969c', '#7b4173', '#a55194','#ce6dbd', '#de9ed6' ] 



def plot_objectives(Populations): 
    leng= len(Populations[0].neuralNetworks[0].accuracyTrain_h)
    fig1, axes = plt.subplots(2,2, figsize=(10,5))
    fig1.suptitle('Test')
    j = 0
    for p in Populations:
        for i in range(leng):
            for n in p.neuralNetworks:
                u = (np.random.rand(4)-0.5)*1.5
                hiddenlayers = int((len(n.layers)- 4)/2)
                volume = hiddenlayers*50
    #            axes[0].scatter(n.accuracyVali_h[i], n.nrNeurons_h[i]+u[1], s=volume, alpha =0.6, color = colors[i])
                if j == 0:
                    axes[0,0].scatter(n.accuracyTest_h[i], n.nrNeurons_h[i]+u[2], s=volume, alpha = 0.6,color = colors[i])
                if j == 1:
                    axes[0,1].scatter(n.accuracyTest_h[i], n.nrNeurons_h[i]+u[2], s=volume, alpha = 0.6,color = colors[i])
                if j == 2:
                    axes[1,0].scatter(n.accuracyTest_h[i], n.nrNeurons_h[i]+u[2], s=volume, alpha = 0.6,color = colors[i])
                if j == 3:
                    axes[1,1].scatter(n.accuracyTest_h[i], n.nrNeurons_h[i]+u[2], s=volume, alpha = 0.6,color = colors[i])
        j+=1
    axes[0,0].set_ylabel('Number of Neurons')
    axes[0,0].set_xlabel('Accuracy Test Data')
    axes[0,0].set_title('Population Size 6')
    axes[1,0].set_ylabel('Number of Neurons')
    axes[1,0].set_xlabel('Accuracy Test Data')
    axes[1,0].set_title('Population Size 18')
    axes[0,1].set_ylabel('Number of Neurons')
    axes[0,1].set_xlabel('Accuracy Test Data')
    axes[0,1].set_title('Population Size 12')
    axes[1,1].set_ylabel('Number of Neurons')
    axes[1,1].set_xlabel('Accuracy Test Data')
    axes[1,1].set_title('Population Size 24')
    plt.show()

def plot_IterationSGD(Populations): 
    fig2, axes = plt.subplots(2,2, figsize=(10,5))
    j = 0
    for p in Populations:
        for n in p.neuralNetworks:
            it = np.array(range(len(n.err_h)))
    #        axes[0,0].plot(it, n.err_h, color = 'midnightblue', alpha = 0.2)
            if j == 0:
                axes[0,0].plot(it, n.accuracyTrain_h_iteration, color = 'midnightblue', alpha = 0.2)    
            if j == 1:
                axes[0,1].plot(it, n.accuracyTrain_h_iteration, color = 'midnightblue', alpha = 0.2)
            if j == 2:
                axes[1,0].plot(it, n.accuracyTrain_h_iteration, color = 'midnightblue', alpha = 0.2)
            if j == 3:
                axes[1,1].plot(it, n.accuracyTrain_h_iteration, color = 'midnightblue', alpha = 0.2)
        j+=1
    axes[0,0].set_ylabel('Accuracy Test Data')
    axes[0,0].set_xlabel('Steps')
    axes[0,0].set_title('Population Size 6')
    axes[1,0].set_ylabel('Accuracy Test Data')
    axes[1,0].set_xlabel('Steps')
    axes[1,0].set_title('Population Size 18')
    axes[0,1].set_ylabel('Accuracy Test Data')
    axes[0,1].set_xlabel('Steps')
    axes[0,1].set_title('Population Size 12')
    axes[1,1].set_ylabel('Accuracy Test Data')
    axes[1,1].set_xlabel('Steps')
    axes[1,1].set_title('Population Size 24')
    
    
# still need to add the average over the top 5 
# different population size 

def plot_testAcc(Populations):
    fig3, axes = plt.subplots(2,2, figsize=(10,5))
    j = 0
    for p in Populations:
        for n in p.neuralNetworks:
            u = (np.random.rand(len(n.accuracyTest_h))-0.5)*0.01
            acc=n.accuracyTest_h+u
            acc= np.insert(acc,0, 0)
            generation = range(len(n.accuracyTest_h)+1)
            generation_ =range(1,len(n.accuracyTest_h)+1)
            if j == 0:
                axes[0,0].plot(generation ,acc , color = 'midnightblue', alpha = 0.2)
            if j == 1:
                axes[0,1].plot(generation ,acc , color = 'midnightblue', alpha = 0.2)
            if j == 2: 
                axes[1,0].plot(generation ,acc , color = 'midnightblue', alpha = 0.2)
            if j == 3:
                axes[1,1].plot(generation ,acc , color = 'midnightblue', alpha = 0.2)
#            u = np.zeros(len(generation_))+(np.random.rand()-0.5)
    #        axes[1].plot(generation_, n.nrNeurons_h+u, color = 'midnightblue', alpha = 0.2)
        average = p.Top5average()
        if j == 0:
            axes[0,0].plot(generation_, average, color= 'midnightblue', linewidth=3)
        if j == 1:
            axes[0,1].plot(generation_, average, color= 'midnightblue', linewidth=3)
        if j == 2:
            axes[1,0].plot(generation_, average, color= 'midnightblue', linewidth=3)
        if j == 3:
            axes[1,1].plot(generation_, average, color= 'midnightblue', linewidth=3)
        j+=1
    axes[0,0].set_ylim([0.5,1])
    axes[0,1].set_ylim([0.5,1])
    axes[1,0].set_ylim([0.5,1])
    axes[1,1].set_ylim([0.5,1])
    axes[0,0].set_ylabel('Accuracy Test Data')
    axes[0,0].set_xlabel('Generations')
    axes[0,0].set_title('Population Size 6')
    axes[1,0].set_ylabel('Accuracy Test Data')
    axes[1,0].set_xlabel('Generations')
    axes[1,0].set_title('Population Size 18')
    axes[0,1].set_ylabel('Accuracy Test Data')
    axes[0,1].set_xlabel('Generations')
    axes[0,1].set_title('Population Size 12')
    axes[1,1].set_ylabel('Accuracy Test Data')
    axes[1,1].set_xlabel('Generations')
    axes[1,1].set_title('Population Size 24')
        
        
            
def plot_exploration(EAs, it): 
    fig4, axes = plt.subplots(2,2, figsize=(10,5))
    j=0
    for ea in EAs: 
        array= np.asarray(ea.exp_nrNeurons_h)
        array = np.reshape(array,(it[-1]+1, (2*ea.popSize)))
        for i in range(ea.popSize): 
            x = (i+1) + np.zeros(2*ea.popSize)
            if j==0:
                axes[0,0].scatter(x, array[i,:], color = 'midnightblue', alpha = 0.7)
            if j==1:
                axes[0,1].scatter(x, array[i,:], color = 'midnightblue', alpha = 0.7)
            if j==2:
                axes[1,0].scatter(x, array[i,:], color = 'midnightblue', alpha = 0.7)
            if j==3:
                axes[1,1].scatter(x, array[i,:], color = 'midnightblue', alpha = 0.7)
        j+=1
    axes[0,0].set_ylabel('Number of Neurons')
    axes[0,0].set_xlabel('Generation')
    
    
    
#def plot_swarm(population):
#    fig5, axes = plt.subplots(2,2, figsize=(10,5))
#    u=0
#    j=0
#    for n in population.neuralNetworks:
#        for i in range(len(n.activationFunctions_hh)): 
#            x2= (i+1) + np.zeros(len(n.activationFunctions_hh[i]))
#            axes[0,0].scatter(x2+u, n.activationFunctions_hh[i], color= colours[j])
#            sns.swarmplot(x2, n.activationFunctions_hh[i], color= colours[j])
#        u+=0.1
#        j+=1
#    plt.show()
#    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





