# support/evaluation_selection.py 


def singleobjective(parent, offspring): 
    for i in range(parent.popSize): 
        if offspring.neuralNetworks[i].err < parent.neuralNetworks[i].err:
            parent.neuralNetworks[i] = offspring.neuralNetworks[i]
            newparent = parent 
    return newparent


def fast_nondominated_sort(population): 
    counter = 1
    front = {}
    front[(counter)] = [] # make a list of directories
    for p in population.neuralNetworks:
        for q in population.neuralNetworks:

            if((p.err < q.err and p.nrNeurons <= q.nrNeurons)or(p.err <= q.err and p.nrNeurons < q.nrNeurons)):
            #if true then p dominates q 
                p.solution.append(q)
            if((p.err > q.err and p.nrNeurons >= q.nrNeurons)or(p.err >= q.err and p.nrNeurons > q.nrNeurons)):
            #if true p is dominated by q
                p.ndominated += 1 #increment counter for p by how many it is dominated
        # if p wasn't dominated by any q then ndominated == 0
        if(p.ndominated == 0):  # then p is a member of first front (best front)
            front[(counter)].append(p)

# so far front 1 has been found
    counter = 1
    while(bool(front[(counter)])): # while list not empty iterate through all networks in front 1,2,... till empty list 
        H = []
        for p in front[(counter)]: # goes trough every element in front list
            for q in p.solution:
                q.ndominated -= 1
                if(q.ndominated==0): # if true the q is member of list H
                    H.append(q)
        counter += 1
        front[(counter)] = H
            
            
            
    
    return front #list


