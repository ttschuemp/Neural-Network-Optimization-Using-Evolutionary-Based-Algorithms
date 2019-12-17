#Plotting_helper.py

import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from matplotlib import cm as CM
import matplotlib.gridspec as gridspec

z = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
#z = [1 ,0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

colors = ['midnightblue', 'darkblue', 'darkslateblue', 'slateblue', 'mediumorchid', 'purple', 'plum', 'orchid', 'pink', 'lavenderblush']


colours =['#393b79','#5254a3', '#6b6ecf', '#9c9ede', '#637939', '#8ca252', '#b5cf6b', '#cedb9c', '#8c6d31', '#bd9e39', '#e7ba52','#e7cb94', '#843c39', '#ad494a', '#d6616b','#e7969c', '#7b4173', '#a55194','#ce6dbd', '#de9ed6' ] 



def plot_objectives(Populations): 
    fig1, axes = plt.subplots(2,2, figsize=(8.2,6.2))
    j = 0
    for p in Populations:
        for n in p.neuralNetworks:
            u = (np.random.rand(4)-0.5)
#            hiddenlayers = int((len(n.layers_h[i])- 4)/2)
            volume = (np.asarray(n.layers_h)+2)**3
            u= (np.random.rand(len(n.nrNeurons_h))-0.5)*0.5
            jitterednr = np.asarray(n.nrNeurons_h) + u
            u2 =(np.random.rand(len(n.accuracyTest_h))-0.5)*0.001
            jitteredacc =np.asarray(n.accuracyTest_h) +u2
#            axes[0].scatter(n.accuracyVali_h[i], n.nrNeurons_h[i]+u[1], s=volume, alpha =0.6, color = colors[i])
            if j == 0:
                axes[0,0].scatter(jitteredacc, jitterednr, s = volume , c = sns.cubehelix_palette(10,reverse=True), alpha=0.8)
            if j == 1: 
                axes[0,1].scatter(jitteredacc, jitterednr, s = volume , c = sns.cubehelix_palette(10,reverse=True), alpha=0.8)
            if j == 2:
                axes[1,0].scatter(jitteredacc, jitterednr, s = volume , c = sns.cubehelix_palette(10,reverse=True), alpha=0.8)
            if j == 3:
                axes[1,1].scatter(jitteredacc, jitterednr, s = volume , c = sns.cubehelix_palette(10,reverse=True), alpha=0.8)
        j+=1
    axes[0,0].set_ylabel('Number of Neurons')
    axes[0,0].set_xlabel('Accuracy Test Set')
    axes[0,0].set_title('Population Size 6')
    axes[1,0].set_ylabel('Number of Neurons')
    axes[1,0].set_xlabel('Accuracy Test Set')
    axes[1,0].set_title('Population Size 14')
    axes[0,1].set_ylabel('Number of Neurons')
    axes[0,1].set_xlabel('Accuracy Test Set')
    axes[0,1].set_title('Population Size 10')
    axes[1,1].set_ylabel('Number of Neurons')
    axes[1,1].set_xlabel('Accuracy Test Set')
    axes[1,1].set_title('Population Size 18')
    axes[0,0].set_xlim([0.85,1.0])
    axes[0,1].set_xlim([0.85,1.0])
    axes[1,0].set_xlim([0.85,1.0])
    axes[1,1].set_xlim([0.85,1.0])
    plt.show()

def plot_IterationSGD(Populations): 
    fig2, axes = plt.subplots(2,2, figsize=(8.2,6.2))
    j = 0
    for p in Populations:
        for n in p.neuralNetworks:
            it = np.array(range(len(n.err_h)))
    #        axes[0,0].plot(it, n.err_h, color = 'midnightblue', alpha = 0.2)
            u = (np.random.rand()-0.5)*0.005
            array = np.asarray(n.accuracyTrain_h_iteration)
            if j == 0:
                axes[0,0].plot(it, array+u, color = 'midnightblue', alpha = 0.3)    
            if j == 1:
                axes[0,1].plot(it, array+u, color = 'midnightblue', alpha = 0.3)
            if j == 2:
                axes[1,0].plot(it, array+u, color = 'midnightblue', alpha = 0.3)
            if j == 3:
                axes[1,1].plot(it, array+u, color = 'midnightblue', alpha = 0.3)
        j+=1
    axes[0,0].set_ylabel('Accuracy Training Set')
    axes[0,0].set_xlabel('Steps')
    axes[0,0].set_title('Population Size 6')
    axes[1,0].set_ylabel('Accuracy Training Set')
    axes[1,0].set_xlabel('Steps')
    axes[1,0].set_title('Population Size 14')
    axes[0,1].set_ylabel('Accuracy Training Set')
    axes[0,1].set_xlabel('Steps')
    axes[0,1].set_title('Population Size 10')
    axes[1,1].set_ylabel('Accuracy Training Set')
    axes[1,1].set_xlabel('Steps')
    axes[1,1].set_title('Population Size 18')
    axes[0,0].set_ylim([0.8,1])
    axes[0,1].set_ylim([0.8,1])
    axes[1,0].set_ylim([0.8,1])
    axes[1,1].set_ylim([0.8,1])

    
    
# still need to add the average over the top 5 
# different population size 

def plot_testAcc(Populations):
    fig3, axes = plt.subplots(2,2, figsize=(8.2,6.2))
    j = 0
    for p in Populations:
        for n in p.neuralNetworks:
            u = (np.random.rand(len(n.accuracyTest_h))-0.5)*0.01
            acc=n.accuracyTest_h+u 
            acc= np.insert(acc,0, 0)
            generation = range(len(n.accuracyTest_h)+1)
            generation_ =range(1,len(n.accuracyTest_h)+1)
            if j == 0:
                axes[0,0].plot(generation ,acc , color = 'midnightblue', alpha = 0.3)
            if j == 1:
                axes[0,1].plot(generation ,acc , color = 'midnightblue', alpha = 0.3)
            if j == 2: 
                axes[1,0].plot(generation ,acc , color = 'midnightblue', alpha = 0.3)
            if j == 3:
                axes[1,1].plot(generation ,acc , color = 'midnightblue', alpha = 0.3)
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
    axes[0,0].set_ylim([0.80,1])
    axes[0,1].set_ylim([0.80,1])
    axes[1,0].set_ylim([0.80,1])
    axes[1,1].set_ylim([0.80,1])
    axes[0,0].set_xlim([1,10])
    axes[0,1].set_xlim([1,10])
    axes[1,0].set_xlim([1,10])
    axes[1,1].set_xlim([1,10])
    axes[0,0].set_ylabel('Accuracy Test Set')
    axes[0,0].set_xlabel('Generations')
    axes[0,0].set_title('Population Size 6')
    axes[1,0].set_ylabel('Accuracy Test Set')
    axes[1,0].set_xlabel('Generations')
    axes[1,0].set_title('Population Size 14')
    axes[0,1].set_ylabel('Accuracy Test Set')
    axes[0,1].set_xlabel('Generations')
    axes[0,1].set_title('Population Size 10')
    axes[1,1].set_ylabel('Accuracy Test Set')
    axes[1,1].set_xlabel('Generations')
    axes[1,1].set_title('Population Size 18')
        
        
            
def plot_exploration(EAs, it): 
    fig4, axes = plt.subplots(2,2, figsize=(8.2,6.2))
    j=0
    for ea in EAs: 
        array= np.asarray(ea.exp_nrNeurons_h)
        array = np.reshape(array,(it, (2*ea.popSize))) # *2 cause its exploration means offspring and population together
        for i in range(it): 
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
    axes[0,0].set_xlabel('Generations')
    axes[0,0].set_title('Population Size 6')
    axes[1,0].set_ylabel('Number of Neurons')
    axes[1,0].set_xlabel('Generations')
    axes[1,0].set_title('Population Size 14')
    axes[0,1].set_ylabel('Number of Neurons')
    axes[0,1].set_xlabel('Generations')
    axes[0,1].set_title('Population Size 10')
    axes[1,1].set_ylabel('Number of Neurons')
    axes[1,1].set_xlabel('Generations')
    axes[1,1].set_title('Population Size 18')
    
    

    
def plot_AD(Populations, it):
    fig5, axes = plt.subplots(1,3, figsize=(12.4,12.4/3))
#    generation = range(1,it+1)
#    a = []
#    for p in Populations:
#        for n in p.neuralNetworks:
#            u = (np.random.rand()-0.5)*2
#            array= np.asarray(n.nrNeurons_h)
##            axes[0].plot(generation, array+u, color= 'midnightblue', alpha = 0.13)
#            u = (np.random.rand()-0.5)*0.1
#            layersh = np.asarray(n.layers_h_generation)
#            axes[2].plot(generation, layersh+u, color= 'midnightblue',alpha = 0.13)
#            d=int(len(n.nrNeurons_h))
#            a += n.nrNeurons_h
#    a = np.asarray(a)
#    a = np.reshape(a,(18*len(Populations),d))
#    average = np.mean(a, axis=0)
##    axes[0].plot(generation, average, color= 'midnightblue', linewidth=3)
#    b=np.zeros(len(a)) *np.nan
#    for i in range(len(a)):
#        b[i]=a[i,-1]
##    axes[1].hist(b, bins = 10, color = 'midnightblue', alpha = 0.9, density=True)
#    bmean = np.mean(b)
#    print(bmean)
#    bstd = np.std(b)
#    print(bstd)
#    bmedian = np.median(b)
#    print(bmedian)
#    quantil= np.quantile(b, [0.1, 0.2, 0.05])
#    print(quantil)
    elitestNN = []
    elitestNN_hid = []
    for p in Populations:
        elitestPop=[]
        for n in p.neuralNetworks:
#            #get all with dominant rank 1
            if n.dominantRank == 1: 
                if n.accuracyTest >= 0.99:
                    elitestPop.append(n)
        elitestPop.sort(key=lambda x: (x.nrNeurons))
        elitestNN.append(elitestPop[0].nrNeurons)
        elitestNN_hid.append(elitestPop[0].layers_h)
    elitestNN.sort()
    axes[1].hist(elitestNN, bins = 10, color = 'midnightblue', alpha = 0.9, density=True)
    elitestNN_arr =  np.asarray(elitestNN)
    leng=len(elitestNN_arr)
    u,c = np.unique(elitestNN_arr, return_counts=True)
    c_cumsum = np.cumsum(c)
    c_cumsum= np.flip(c_cumsum)
    total= np.zeros(len(u))+leng
    u= np.flip(u)
    c_cumsum_2=c_cumsum/total
    axes[0].plot(u, c_cumsum_2, color = 'midnightblue', alpha = 0.9,linewidth=3)
    axes[0].set_xlabel('Number of Neurons')
    axes[0].set_ylabel('Cumulative Probability')
    axes[1].set_xlabel('Number of Neurons')
    axes[1].set_ylabel('Probability Density')
    axes[2].set_ylabel('Hiddenlayers')
    axes[2].set_xlabel('Generations')
    
    
    
def plot_AD3(Populations, it):
    fig6 = plt.figure(constrained_layout=True)
    elitestNN = []
    elitestNN_hid = []
    for p in Populations:
        elitestPop=[]
        for n in p.neuralNetworks:
#            #get all with dominant rank 1
            if n.dominantRank == 1: 
                if n.accuracyTest >= 0.99:
                    elitestPop.append(n)
        elitestPop.sort(key=lambda x: (x.nrNeurons))
        elitestNN.append(elitestPop[0].nrNeurons)
        elitestNN_hid.append(elitestPop[0].layers_h)
    elitestNN.sort()
    axes[1].hist(elitestNN, bins = 10, color = 'midnightblue', alpha = 0.9, density=True)
    elitestNN_arr =  np.asarray(elitestNN)
    leng=len(elitestNN_arr)
    u,c = np.unique(elitestNN_arr, return_counts=True)
    c_cumsum = np.cumsum(c)
    c_cumsum= np.flip(c_cumsum)
    total= np.zeros(len(u))+leng
    u= np.flip(u)
    c_cumsum_2=c_cumsum/total
    axes[0].plot(u, c_cumsum_2, color = 'midnightblue', alpha = 0.9,linewidth=3)
    axes[0].set_xlabel('Number of Neurons')
    axes[0].set_ylabel('Cumulative Probability')
    axes[1].set_xlabel('Number of Neurons')
    axes[1].set_ylabel('Probability Density')
    axes[2].set_ylabel('Hiddenlayers')
    axes[2].set_xlabel('Generations')
    
    gs = gridspec.GridSpec(1, 3)
    f6_ax1 = fig6.add_subplot(gs[0:2])
    f6_ax2 = fig6.add_subplot(gs[2:3])




   #     [30, 21,18,10,9,8,6]    [10/10, 9/10, 8/10,7/10,6/10,3/10,2/10]
#N = 50  6,8,9,10,18,21,30]
#x = np.arange(N)
## Here are many sets of y to plot vs x
#ys = [x + i for i in x]
#
## We need to set the plot limits, they will not autoscale
#fig, ax = plt.subplots()
#ax.set_xlim(np.min(x), np.max(x))
#ax.set_ylim(np.min(ys), np.max(ys))
#
#line_segments = LineCollection([np.column_stack([x, y]) for y in ys],
#                               linewidths=(0.5, 1, 1.5, 2),
#                               linestyles='solid')
#line_segments.set_array(x)
#ax.add_collection(line_segments)
#axcb = fig.colorbar(line_segments)
#axcb.set_label('Line Number')
#ax.set_title('Line Collection with mapped colors')
#plt.sci(line_segments)  # This allows interactive changing of the colormap.
#plt.show()




