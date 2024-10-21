import random
from copy import deepcopy
from genopt.genom import Categorial, Discrete, Integer, Continues
from .utils import flip_coin


def arithmetic_breed(g1, g2):
    alpha = random.random()
    return alpha * g1 + (1 - alpha) * g2


def permutation_crossover(mother, father, space):

    sister = deepcopy(mother)
    brother = deepcopy(father)
    
    for gen in mother.gens():
        if isinstance(space[gen], (Categorial, Discrete)):
            if flip_coin(0.5):
                sister[gen] = father[gen]
                brother[gen] = mother[gen]

        elif isinstance(space[gen], Integer):
            sister[gen] = round(arithmetic_breed(mother[gen], father[gen]))
            brother[gen] = round(arithmetic_breed(mother[gen], father[gen]))

        else:
            sister[gen] = arithmetic_breed(mother[gen], father[gen])
            brother[gen] = arithmetic_breed(mother[gen], father[gen])
    
    return sister, brother