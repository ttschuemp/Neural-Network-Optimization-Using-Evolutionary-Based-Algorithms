# NSGAII.py 

import numpy as np
from Population import Population
from support.Evaluation_Selection import fast_nondominated_sort, crowding_distance_assignment


class NSGAII: 

    def run(self, parent, offspring): 
        lenPop = parent.popSize
        combPop = Population(parent.neuralNetworks + offspring.neuralNetworks) # combine the two populations
        
        front = fast_nondominated_sort(combPop)
#        if len(front) == 0:
#            return "front to small"
        # add first front until > popSize
        newParentList = []
        counter = 1
        while(len(newParentList) < lenPop):
            for f in front[(counter)]: # key error cause front two small 
                newParentList.append(f)
            counter += 1
            if(len(newParentList) > lenPop):
                break
        counter -= 1
        crowding_distance_assignment(front[(counter)]) #distance of each point is set front[(counter)]
        # pick the first deltaN points
        #sort newParentList
        newParentList.sort(reverse=True, key=lambda x: (-x.dominantRank, x.crowdingDistance))
        newParentList = newParentList[0:lenPop]
        #make new population
        newParentPop = Population(newParentList)
            
        return newParentPop

            
        
        
        
        
