from genopt.genopt_algorithm import SGA
from genopt.scalers import sigma_trunc_scaling
from genopt.selector import tournament_selection
from genopt.crossovers import permutation_crossover
from genopt.mutators import uniform_mutation


def genetic_optimize(objective, param_grid, n_iter=10):
    
    ga = SGA(task='minimize', pop_size=6, cross_prob=0.8, mut_prob=0.1, elitism=True)
    ga.set_selector_type(tournament_selection)
    ga.set_scaler_type(sigma_trunc_scaling)
    ga.set_crossover_type(permutation_crossover)
    ga.set_mutator_type(uniform_mutation)
    ga.set_fitness(objective)
    print(1)
    ga.initialize_pop(param_grid)
    
    for i in range(n_iter):
        print(2)
        ga.step()
        
    best_params = ga.best_individual().container
    
    return best_params