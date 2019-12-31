# NSGAII.py 

import numpy as np
from Population import Population
from support.Evaluation_Selection import fast_nondominated_sort, crowding_distance_assignment


class NSGAII: 

    def run(self, parent, offspring): 
        lenPop = parent.popSize
        combPop = Population(parent.neuralNetworks + offspring.neuralNetworks) # combine the two populations
        front = fast_nondominated_sort(combPop) # sorts NN into fronts (rank 1 is best = first fornt)
        newParentList = []
        counter = 1
        while(len(newParentList) < lenPop): # append fronts to newParentList
            for f in front[(counter)]: 
                newParentList.append(f)
            counter += 1
            if(len(newParentList) > lenPop):
                break
        counter -= 1
        crowding_distance_assignment(front[(counter)]) #calculates distance of each NN point in objective space
        #pick the first deltaN points
        #sort newParentList
        newParentList.sort(reverse=True, key=lambda x: (-x.dominantRank, x.crowdingDistance))
        newParentList = newParentList[0:lenPop]
        #make new population
        newParentPop = Population(newParentList)
            
        return newParentPop

            
        
        
        
        
