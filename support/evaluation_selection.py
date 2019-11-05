# support/Evaluation_Selection.py 


def singleobjective(parent, offspring): 
    for i in range(parent.popSize): 
        if offspring.neuralNetworks[i].err < parent.neuralNetworks[i].err:
            parent.neuralNetworks[i] = offspring.neuralNetworks[i]
            newparent = parent 
        else:
            return parent
    return newparent


def getLofDirecList(direc):
    lgth = 0
    for d in range(1,len(direc)+1):
       lgth += len(direc[d])
    return lgth


def fast_nondominated_sort(population): #ranks population
    size = (population.popSize/2) # final parent population size 
    counter = 1
    front = {}
    front[(counter)] = [] # make a list of directories
    for p in population.neuralNetworks:
        for q in population.neuralNetworks:

            if((p.accuracyOOS > q.accuracyOOS and p.nrNeurons <= q.nrNeurons)or(p.accuracyOOS >= q.accuracyOOS and p.nrNeurons < q.nrNeurons)):
            #if true then p dominates q 
                p.solution.append(q)
            if((p.accuracyOOS < q.accuracyOOS and p.nrNeurons >= q.nrNeurons)or(p.accuracyOOS <= q.accuracyOOS and p.nrNeurons > q.nrNeurons)):
            #if true p is dominated by q
                p.ndominated += 1 #increment counter for p by how many it is dominated
        # if p wasn't dominated by any q then ndominated == 0
        if(p.ndominated == 0):  # then p is a member of first front (best front)
            front[(counter)].append(p)
            p.dominantRank = counter 

# so far front 1 has been found
    counter = 1
    while(bool(front[(counter)])): # while list not empty iterate through all networks in front 1,2,... till empty list 
        H = []
        for p in front[(counter)]: # goes trough every element in front list
            for q in p.solution:
                q.ndominated -= 1
                if(q.ndominated==0): # if true the q is member of list H
                    H.append(q)
            p.solution = []
        counter += 1
        front[(counter)] = H
        for t in front[(counter)]:
            t.dominantRank = counter
            
    # size of front to small to get a new population? 
       
#    lgth = getLofDirecList(front)
#    print("front länge:",lgth)
#    if(lgth < size):
#        front = []
#        return front 
            
        
    return front #list

# help functions for sort()

def takeErr(elem):
    
    return elem.accuracyOOS

def takeNrNeurons(elem):
    
    return elem.nrNeurons

#

def crowding_distance_assignment(setI): 
    length = len(setI)
    for i in setI: 
        i.crowdingDistance = 0 # initialise crowding distance with 0 
    setI.sort(key = takeErr) #sort list according to error
    setI[0].crowdingDistance = 10e+6 #inf 
    setI[-1].crowdingDistance = 10e+6 #inf 
    # for all other points
    for j in range(1, length-1): 
        setI[j].crowdingDistance += (setI[j+1].accuracyOOS - setI[j-1].accuracyOOS)
        
    setI.sort(key = takeNrNeurons) #sort list according to nrNeurons
    setI[0].crowdingDistance = 10e+6
    setI[-1].crowdingDistance = 10e+6
    for j in range(1, length-1): 
        setI[j].crowdingDistance += (setI[j+1].nrNeurons - setI[j-1].nrNeurons)/setI[0].maxNeurons # divide by max number...
        #of neurons such that neurons have not higher impact
        
        
        
    
    
    


