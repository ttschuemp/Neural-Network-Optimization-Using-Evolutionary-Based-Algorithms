#Plotting_helper.py

import numpy as np
import matplotlib
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from matplotlib import cm as CM
import matplotlib.gridspec as gridspec



def plot_objectives_2(Populations): 
    fig1, axes = plt.subplots(2,2, figsize=(8.2,6.2))
    i=0.0
    g = lambda s: s**(1/3)-2
    for n in Populations[0].neuralNetworks:
        nrNeurons_array = np.asarray(n.nrNeurons_h)
        nrNeurons_array = nrNeurons_array+i
        i+=0.1
        scatter1=axes[0,0].scatter(n.accuracyTest_h, nrNeurons_array,linewidths=0.3, edgecolors= 'w', s = (np.asarray(n.layers_h_generation)+2)**3 , c = [1,2,3,4,5,6,7,8,9,10],cmap = sns.cubehelix_palette(10,reverse=True, as_cmap = True) , alpha=0.8)
        handles, labels = scatter1.legend_elements(prop="sizes", alpha=0.90, func=g)
    c=fig1.colorbar(scatter1, ax=axes[0,0])
    c.set_label('Generations')
    legend1 = axes[0,0].legend(handles, labels, loc="best", title="Hidden Layers",frameon=True, fancybox=True, framealpha=0.4)
    axes[0,0].add_artist(legend1)
    i=0.0
    for n in Populations[2].neuralNetworks:
        nrNeurons_array = np.asarray(n.nrNeurons_h)
        nrNeurons_array =nrNeurons_array+ i
        i+=0.1
        scatter3=axes[1,0].scatter(n.accuracyTest_h, nrNeurons_array,linewidths=0.3, edgecolors= 'w', s = (np.asarray(n.layers_h_generation)+2)**3 , c = [1,2,3,4,5,6,7,8,9,10], cmap = sns.cubehelix_palette(10,reverse=True, as_cmap = True) , alpha=0.8)
        handles2, labels2 = scatter3.legend_elements(prop="sizes", alpha=0.90,func=g)
    c2=fig1.colorbar(scatter3, ax=axes[1,0])
    c2.set_label('Generations')
    legend3 = axes[1,0].legend(handles2, labels2, loc="best", title="Hidden Layers",frameon=True, fancybox=True, framealpha=0.4)
    axes[1,0].add_artist(legend3)

    i=0.0
    for n in Populations[1].neuralNetworks:
        nrNeurons_array = np.asarray(n.nrNeurons_h)
        nrNeurons_array =nrNeurons_array+ i
        i+=0.1
        scatter2=axes[0,1].scatter(n.accuracyTest_h, nrNeurons_array,linewidths=0.3, edgecolors= 'w', s = (np.asarray(n.layers_h_generation)+2)**3 , c = [1,2,3,4,5,6,7,8,9,10], cmap = sns.cubehelix_palette(10,reverse=True, as_cmap = True) , alpha=0.8)
        handles1, labels1 = scatter2.legend_elements(prop="sizes", alpha=0.90,func=g)
    c1=fig1.colorbar(scatter2, ax=axes[0,1])
    c1.set_label('Generations')
    legend2 = axes[0,1].legend(handles2, labels2, loc="best", title="Hidden Layers", frameon=True, fancybox=True, framealpha=0.4)
    axes[0,1].add_artist(legend2)

    i=0.0

    for n in Populations[3].neuralNetworks:
        nrNeurons_array = np.asarray(n.nrNeurons_h)
        nrNeurons_array =nrNeurons_array+ i
        i+=0.1
        scatter4=axes[1,1].scatter(n.accuracyTest_h, nrNeurons_array,linewidths=0.3, edgecolors= 'w', s = (np.asarray(n.layers_h_generation)+2)**3 , c = [1,2,3,4,5,6,7,8,9,10], cmap = sns.cubehelix_palette(10,reverse=True, as_cmap = True) , alpha=0.8)
        handles3, labels3 = scatter4.legend_elements(prop="sizes", alpha=0.90,func=g)
    legend4 = axes[1,1].legend(handles2, labels2, loc="best", title="Hidden Layers",frameon=True, fancybox=True, framealpha=0.4)
    axes[1,1].add_artist(legend4)
    c3=fig1.colorbar(scatter4, ax=axes[1,1])
    c3.set_label('Generations')

    axes[0,0].set_ylabel('Number of Neurons')
    axes[0,0].set_xlabel('Accuracy Test Sample')
    axes[0,0].set_title('Population Size 6')
    axes[1,0].set_ylabel('Number of Neurons')
    axes[1,0].set_xlabel('Accuracy Test Sample')
    axes[1,0].set_title('Population Size 18')
    axes[0,1].set_ylabel('Number of Neurons')
    axes[0,1].set_xlabel('Accuracy Test Sample')
    axes[0,1].set_title('Population Size 12')
    axes[1,1].set_ylabel('Number of Neurons')
    axes[1,1].set_xlabel('Accuracy Test Sample')
    axes[1,1].set_title('Population Size 24')
    axes[0,0].set_xlim([0.80,0.95])
    axes[0,1].set_xlim([0.80,0.95])
    axes[1,0].set_xlim([0.80,0.95])
    axes[1,1].set_xlim([0.80,0.95])
    axes[0,0].set_ylim([800,1600])
    axes[0,1].set_ylim([800,1600])
    axes[1,0].set_ylim([800,1600])
    axes[1,1].set_ylim([800,1600])

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
    axes[0,0].set_ylabel('Accuracy Training Sample')
    axes[0,0].set_xlabel('Steps')
    axes[0,0].set_title('Population Size 6')
    axes[1,0].set_ylabel('Accuracy Training Sample')
    axes[1,0].set_xlabel('Steps')
    axes[1,0].set_title('Population Size 18')
    axes[0,1].set_ylabel('Accuracy Training Sample')
    axes[0,1].set_xlabel('Steps')
    axes[0,1].set_title('Population Size 12')
    axes[1,1].set_ylabel('Accuracy Training Sample')
    axes[1,1].set_xlabel('Steps')
    axes[1,1].set_title('Population Size 24')
    axes[0,0].set_ylim([0.86,1])
    axes[0,1].set_ylim([0.86,1])
    axes[1,0].set_ylim([0.86,1])
    axes[1,1].set_ylim([0.86,1])
    axes[0,0].set_xlim([0,80])
    axes[0,1].set_xlim([0,80])
    axes[1,0].set_xlim([0,80])
    axes[1,1].set_xlim([0,80])


    
    
# still need to add the average over the top 5 
# different population size 

def plot_testAcc(Populations):
    fig3, axes = plt.subplots(2,2, figsize=(8.2,6.2))
    j = 0
    for p in Populations:
        for n in p.neuralNetworks:
            u = (np.random.rand()-0.5)*0.01
            acc=np.asarray(n.accuracyTest_h)+u 
#            acc= np.insert(acc,0, 0)
#            generation = range(len(n.accuracyTest_h)+1)
            generation_ =range(1,len(n.accuracyTest_h)+1)
            if j == 0:
                axes[0,0].plot(generation_ ,acc , color = 'midnightblue', alpha = 0.3)
            if j == 1:
                axes[0,1].plot(generation_ ,acc , color = 'midnightblue', alpha = 0.3)
            if j == 2: 
                axes[1,0].plot(generation_ ,acc , color = 'midnightblue', alpha = 0.3)
            if j == 3:
                axes[1,1].plot(generation_ ,acc , color = 'midnightblue', alpha = 0.3)
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
    axes[0,0].set_ylim([0.80,0.96])
    axes[0,1].set_ylim([0.80,0.96])
    axes[1,0].set_ylim([0.80,0.96])
    axes[1,1].set_ylim([0.80,0.96])
    axes[0,0].set_xlim([1,10])
    axes[0,1].set_xlim([1,10])
    axes[1,0].set_xlim([1,10])
    axes[1,1].set_xlim([1,10])
    axes[0,0].set_ylabel('Accuracy Test Sample')
    axes[0,0].set_xlabel('Generations')
    axes[0,0].set_title('Population Size 6')
    axes[1,0].set_ylabel('Accuracy Test Sample')
    axes[1,0].set_xlabel('Generations')
    axes[1,0].set_title('Population Size 18')
    axes[0,1].set_ylabel('Accuracy Test Sample')
    axes[0,1].set_xlabel('Generations')
    axes[0,1].set_title('Population Size 12')
    axes[1,1].set_ylabel('Accuracy Test Sample')
    axes[1,1].set_xlabel('Generations')
    axes[1,1].set_title('Population Size 24')
        
        
            
def plot_exploration(EAs, it): 
    fig4, axes = plt.subplots(2,2, figsize=(8.2,6.2))
    j=0
    for ea in EAs: 
        array= np.asarray(ea.exp_nrNeurons_h)
        array = np.reshape(array,(it, (2*ea.popSize))) # *2 cause its exploration means offspring and population together
        for i in range(it): 
            x = (i+1) + np.zeros(2*ea.popSize)
            if j==0:
                axes[0,0].scatter(x, array[i,:], color = (0.368, 0.217, 0.415), alpha = 0.9)
            if j==1:
                axes[0,1].scatter(x, array[i,:], color = (0.368, 0.217, 0.415), alpha = 0.9)
            if j==2:
                axes[1,0].scatter(x, array[i,:], color = (0.368, 0.217, 0.415), alpha = 0.9)
            if j==3:
                axes[1,1].scatter(x, array[i,:], color= (0.368, 0.217, 0.415), alpha = 0.9)
        j+=1
    axes[0,0].set_ylabel('Number of Neurons')
    axes[0,0].set_xlabel('Generations')
    axes[0,0].set_title('Population Size 6')
    axes[1,0].set_ylabel('Number of Neurons')
    axes[1,0].set_xlabel('Generations')
    axes[1,0].set_title('Population Size 18')
    axes[0,1].set_ylabel('Number of Neurons')
    axes[0,1].set_xlabel('Generations')
    axes[0,1].set_title('Population Size 12')
    axes[1,1].set_ylabel('Number of Neurons')
    axes[1,1].set_xlabel('Generations')
    axes[1,1].set_title('Population Size 24')
    axes[0,0].set_ylim([800,2200])
    axes[0,1].set_ylim([800,2200])
    axes[1,0].set_ylim([800,2200])
    axes[1,1].set_ylim([800,2200])
    

    
#def plot_AD(Populations, it):
#    fig5, axes = plt.subplots(1,3, figsize=(12.4,12.4/3))
##    generation = range(1,it+1)
##    a = []
##    for p in Populations:
##        for n in p.neuralNetworks:
##            u = (np.random.rand()-0.5)*2
##            array= np.asarray(n.nrNeurons_h)
###            axes[0].plot(generation, array+u, color= 'midnightblue', alpha = 0.13)
##            u = (np.random.rand()-0.5)*0.1
##            layersh = np.asarray(n.layers_h_generation)
##            axes[2].plot(generation, layersh+u, color= 'midnightblue',alpha = 0.13)
##            d=int(len(n.nrNeurons_h))
##            a += n.nrNeurons_h
##    a = np.asarray(a)
##    a = np.reshape(a,(18*len(Populations),d))
##    average = np.mean(a, axis=0)
###    axes[0].plot(generation, average, color= 'midnightblue', linewidth=3)
##    b=np.zeros(len(a)) *np.nan
##    for i in range(len(a)):
##        b[i]=a[i,-1]
###    axes[1].hist(b, bins = 10, color = 'midnightblue', alpha = 0.9, density=True)
##    bmean = np.mean(b)
##    print(bmean)
##    bstd = np.std(b)
##    print(bstd)
##    bmedian = np.median(b)
##    print(bmedian)
##    quantil= np.quantile(b, [0.1, 0.2, 0.05])
##    print(quantil)
#    elitestNN = []
#    elitestNN_hid = []
#    for p in Populations:
#        elitestPop=[]
#        for n in p.neuralNetworks:
##            #get all with dominant rank 1
#            if n.dominantRank == 1: 
#                if n.accuracyTest >= 0.99:
#                    elitestPop.append(n)
#        elitestPop.sort(key=lambda x: (x.nrNeurons))
#        elitestNN.append(elitestPop[0].nrNeurons)
#        elitestNN_hid.append(elitestPop[0].layers_h)
#    elitestNN.sort()
#    axes[1].hist(elitestNN, bins = 10, color = 'midnightblue', alpha = 0.9, density=True)
#    elitestNN_arr =  np.asarray(elitestNN)
#    leng=len(elitestNN_arr)
#    u,c = np.unique(elitestNN_arr, return_counts=True)
#    c_cumsum = np.cumsum(c)
#    c_cumsum= np.flip(c_cumsum)
#    total= np.zeros(len(u))+leng
#    u= np.flip(u)
#    c_cumsum_2=c_cumsum/total
#    axes[0].plot(u, c_cumsum_2, color = 'midnightblue', alpha = 0.9,linewidth=3)
#    axes[0].set_xlabel('Number of Neurons')
#    axes[0].set_ylabel('Cumulative Probability')
#    axes[1].set_xlabel('Number of Neurons')
#    axes[1].set_ylabel('Probability Density')
#    axes[2].set_ylabel('Hiddenlayers')
#    axes[2].set_xlabel('Generations')
#    
    
    
def plot_AD(Populations, it):
    fig6 = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(1, 4)
    f6_ax1 = fig6.add_subplot(gs[0:3])
    f6_ax2 = fig6.add_subplot(gs[3:4])
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
    f6_ax2.hist(elitestNN, bins = 6, color = 'midnightblue', alpha = 0.9, density=True)
    elitestNN_arr =  np.asarray(elitestNN)
    leng=len(elitestNN_arr)
    u,c = np.unique(elitestNN_arr, return_counts=True)
    c_cumsum = np.cumsum(c)
    c_cumsum= np.flip(c_cumsum)
    total= np.zeros(len(u))+leng
    u= np.flip(u)
    c_cumsum_2=c_cumsum/total
    f6_ax1.plot(u, c_cumsum_2, color = 'midnightblue', alpha = 0.9,linewidth=3)
    f6_ax1.set_xlabel('Number of Neurons')
    f6_ax1.set_ylabel('Cumulative Probability')
    f6_ax2.set_xlabel('Number of Neurons')
    f6_ax2.set_ylabel('Probability Density')

    
#    # Plot of AD Training Dataset
#    plt.figure(figsize=(3,3))
#    plt.scatter(X_train[:,0],X_train[:,1], c=np.argmax(y_train, axis = 1), cmap=matplotlib.colors.ListedColormap(sns.cubehelix_palette(dark=0.3, light=0.5)), s=55, alpha=0.92)
#    plt.scatter(X_test[:,0],X_test[:,1], c=np.argmax(y_test,axis = 1), cmap=matplotlib.colors.ListedColormap(sns.cubehelix_palette(dark=0.3, light=0.5)),s=55, alpha=0.92)
#    plt.ylim(0,1) 
#    plt.xlim(0,1)
#    plt.show()


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




