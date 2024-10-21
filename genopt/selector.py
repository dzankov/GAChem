import random
from math import ceil
	
def roulette_wheel_selection(pop):
    
    if pop.minimax == 'maximize':
        sumFit = sum(ind.fitness for ind in pop)
        weights = [ind.fitness / sumFit for ind in pop]
    else:
        fitMax = pop.stats['fitMax']
        sumFit = sum(fitMax - ind.fitness for ind in pop)
        
        if sumFit == 0: 
            fitMax = 1 / len(pop)
            sumFit = 1
        
        weights = [(fitMax - ind.fitness) / sumFit for ind in pop]
        
    probs = [sum(weights[:i + 1]) for i in range(len(weights))]
    
    selectedInds = []
    for i in range(len(pop)):
        pick = random.random()
        for i in range(len(pop)):
            if pick <= probs[i]:
                selectedInds.append(i)
                break


    return selectedInds

	
def linear_rank_selection(pop):
    
    pop.ranking()
    
    sumRank = sum(ind.rank for ind in pop)
    weights = [ind.rank / sumRank for ind in pop]
    probs = [sum(weights[:i + 1]) for i in range(len(weights))]
    
    selectedInds = []
    for i in range(len(pop)):
        pick = random.random()
        for i in range(len(pop)):
            if pick <= probs[i]:
                selectedInds.append(i)
                break

    return selectedInds
	

def tournament_selection(pop):
    
    selectedInds = []
    for i in range(len(pop)):
        p1 = random.randint(0, len(pop) - 1)
        p2 = random.randint(0, len(pop) - 1)
        
        if pop.minimax == 'maximize':
            selectedInds.append(max(p1, p2, key=lambda x: pop[x].score))
        else:
            selectedInds.append(min(p1, p2, key=lambda x: pop[x].score))
            
    return selectedInds
	
