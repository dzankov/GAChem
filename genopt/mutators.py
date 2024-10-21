import random
from copy import deepcopy
from .utils import flip_coin


def uniform_mutation(individual, space, prob=0.01):
    
    mutant = deepcopy(individual)
    
    for gen in mutant.gens():
        if flip_coin(prob):
            mutant[gen] = space[gen].get_point()
            
    return mutant