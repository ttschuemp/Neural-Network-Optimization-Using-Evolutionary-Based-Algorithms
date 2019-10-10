# support/evaluation_selection.py 


def singleobjective(parent, offspring): 
    for i in range(parent.popSize): 
        if offspring.neuralNetworks[i].err < parent.neuralNetworks[i].err:
            parent.neuralNetworks[i] = offspring.neuralNetworks[i]
            newparent = parent 
    return newparent