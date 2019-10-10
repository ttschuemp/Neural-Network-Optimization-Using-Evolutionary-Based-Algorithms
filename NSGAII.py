# NSGAII.py 

import numpy as np
from Population import Population


class NSGAII: 
    
    
    def __init__(self, parent, offspring): 
        self.parent = parent
        self.offspring = offspring
        self.combPop = Population(self.parent.neuralNetworks + self.offspring.neuralNetworks) # combine the two populations
        
        
    def run(self): 
        
        Front = fast_nondominated_sort(self.combPop)
