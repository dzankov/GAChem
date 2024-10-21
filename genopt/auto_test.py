import os
import time
import numpy as np
from functools import reduce
from operator import mul
from math  import ceil
from statistics import mean
from scipy.integrate import simps
from itertools import groupby, chain
from datetime import datetime
from genopt.genopt_algorithm import SGA


def genalg_preparer(gen_operators):
    
    gen_algs = []
    
    for selector in gen_operators['selectors']:
        for scaler in gen_operators['scalers']:
            for crossover in gen_operators['crossovers']:
                for mutator in gen_operators['mutators']:

                    ga = SGA(task='minimize', pop_size=10, cross_prob=0.8, mut_prob=0.1, elitism=True)

                    ga.set_selector_type(selector)
                    ga.set_scaler_type(scaler)
                    ga.set_crossover_type(crossover)
                    ga.set_mutator_type(mutator)

                    gen_algs.append(ga)
                    
    gen_algs_wrapped = [GenAlgWrapper(alg) for alg in gen_algs]
                    
    return gen_algs_wrapped

def calc_n_iter(space_size, pop_size=10, rate_coef=10):
    n_iter = ceil(space_size / pop_size / rate_coef) - 1

    if n_iter < 1:
        return 1
    return n_iter


class AlgWrapper:

    def __init__(self, trials=10):
        self.trials = trials
        self.top_three = 0
        self.first = 0
        self.borda = 0
   
        self.meta = {}
        
    
    def auc_calc(self, auc_list):
        auc_list_scaled = self.min_max_scaling(auc_list)
        auc = simps(auc_list_scaled, range(len(auc_list_scaled)))
        return auc
    
   
    def min_max_scaling(self, x):
        x_min = min(x)
        x_max = max(x)
        if x_min == x_max:
            return x
        x_scaled = [(i - x_min) / (x_max - x_min) for i in x]
        return x_scaled
    

class HyperOptWrapper(AlgWrapper):
    
    def __init__(self, alg, trials=10):
        super().__init__()
        self.alg = alg
    
    
    def add_test_func(self, func):
        self.alg.set_fitness(func)
        self.fname = func.__name__.lower()
        self.meta[self.fname] = None
        
    
    def initialize(self):
        self.param_grid = self.alg.fitness.param_grid_hyperopt
        #self.n_iter = 10 * (calc_n_iter(self.alg.fitness().comb, pop_size=10, rate_coef=10) + 1)
        self.n_iter = 1
    
    def fit(self):
        
        stats = {'BestRank':0,
                 'AucRank':0,
                 'Catch':0,
                 'TimeList':[],
                 'BestList':[],
                 'AucList':[],
                 'R2List':[],
                 'RmseList':[]
                }

        for trial in range(self.trials):
            start = time.time()
            
            self.alg.initialize(self.param_grid)
            self.alg.fit(n_iter=self.n_iter)
            
            auc_tmp = self.alg.loss_scores()
            
            stats['TimeList'].append(time.time() - start)
            stats['BestList'].append(self.alg.best_score())
            stats['AucList'].append(self.auc_calc(auc_tmp))
            
            r2, rmse = self.alg.fitness().external_test(self.alg.best_solution())
            stats['R2List'].append(r2)
            stats['RmseList'].append(rmse)
        
        if self.alg.fitness().min_loc == self.alg.best_solution():
                stats['Catch'] += 1
    
        
        stats['BestAvg'] = mean(stats['BestList'])
        stats['AucAvg'] = mean(stats['AucList'])
        stats['R2Avg'] = mean(stats['R2List'])
        stats['RmseAvg'] = mean(stats['RmseList'])
        stats['TimeAvg'] = mean(stats['TimeList'])
        stats['Steps'] = self.n_iter
        stats['fmin'] = self.alg.fitness().fmin
        stats['comb'] = self.alg.fitness().comb
        stats['R2max'] = self.alg.fitness().r2
        stats['RMSEmin'] = self.alg.fitness().r2
		
        self.meta[self.fname] = stats
        
        return self
            
    
    def __repr__(self):
        return '{};{};;'.format('HyperOpt', self.alg.algo)

        
class GenAlgWrapper(AlgWrapper):
    
    def __init__(self, genalg):
        super().__init__()
        self.genalg = genalg
    
    
    def add_test_func(self, func):
        self.test_func = func
    
    
    def initialize(self):
    
        self.param_grid = self.test_func.param_grid_genopt
        #self.n_iter = calc_n_iter(self.test_func().comb, self.genalg.pop_size, rate_coef=10)
        self.n_iter = 1
        self.fname = self.test_func.__name__.lower()
        self.meta[self.fname] = None 
        
        def func(params):
            model = self.test_func()
            return model.evaluate(params.container)
        
        func.__name__ = self.fname
        self.genalg.set_fitness(func)

    
    def fit(self):
        
        stats = {'BestRank':0,
                 'AucRank':0,
                 'Catch':0,
                 'TimeList':[],
                 'BestList':[],
                 'AucList':[],
                 'R2List':[],
                 'RmseList':[]
                }

        for trial in range(self.trials):
            start = time.time()

            self.genalg.initialize_pop(self.param_grid)
            auc_tmp = []
            
            for i in range(self.n_iter):
                self.genalg.step()
                auc_tmp.append(self.genalg.best_individual().score)

            stats['TimeList'].append(time.time() - start)
            stats['BestList'].append(self.genalg.best_individual().score)
            stats['AucList'].append(self.auc_calc(auc_tmp))
            
            r2, rmse = self.test_func().external_test(self.genalg.best_individual().container)
            stats['R2List'].append(r2)
            stats['RmseList'].append(rmse)
            
            if self.test_func().min_loc == self.genalg.best_individual().container:
                stats['Catch'] += 1
            
        
        stats['BestAvg'] = mean(stats['BestList'])
        stats['AucAvg'] = mean(stats['AucList'])
        stats['R2Avg'] = mean(stats['R2List'])
        stats['RmseAvg'] = mean(stats['RmseList'])
        stats['TimeAvg'] = mean(stats['TimeList'])
        stats['Steps'] = len(self.genalg.cemetery)
        stats['fmin'] = self.test_func().fmin
        stats['comb'] = self.test_func().comb
        stats['R2max'] = self.test_func().r2
        stats['RMSEmin'] = self.test_func().rmse
        
        self.meta[self.fname] = stats
        
        return self
        
    
    def __repr__(self):
        
        sel = self.genalg.selector.__name__
        scal = self.genalg.scaler.__name__
        cross = self.genalg.crossover.__name__
        mut = self.genalg.mutator.__name__
        
        return '{};{};{};{}'.format(sel, scal, cross, mut)
        
    
class AlgAgregator:
    
    def __init__(self):
        pass
        
    
    def add_algs(self, algs):
        self.algs = algs
       
    
    def add_test_funcs(self, test_funcs):
        self.test_funcs = test_funcs
        

    def u_test(self, A, B):
    
        n = len(A)
        sample = [[1, v] for v in A] + [[2, v] for v in B]
        sample.sort(key=lambda x: x[1])

        for rank, i in enumerate(sample, 1):
            i.append(rank)

        R1_sum = sum(i[2] for i in sample if i[0] == 1)
        R2_sum = sum(i[2] for i in sample if i[0] == 2)
        
        Un = n * n + n * (n + 1) / 2
        U = min(Un - R1_sum, Un - R2_sum)

        if U <= 27:
            if mean(A) < mean(B):
                return 1
            else:
                return -1

        return 0
        
    
    def fit(self):
    
        calcs = len(self.test_funcs) * len(self.algs)
        n = 0
        for func in self.test_funcs:
            for alg in self.algs:
                alg.add_test_func(func)
                alg.initialize()
                alg.fit()
                n += 1
                with open('Calculations.csv', 'w') as f:
                    f.write('{:.0f}'.format(100 * n/calcs))
                
        return self
    
    
    def ranks_calc(self):
        
        for func in self.algs[0].meta:
            for alg_A in self.algs:
                for alg_B in self.algs:
                    best_res = self.u_test(alg_A.meta[func]['BestList'], alg_B.meta[func]['BestList'])
                    auc_res = self.u_test(alg_A.meta[func]['AucList'], alg_B.meta[func]['AucList'])

                    if best_res == 1:
                        alg_A.meta[func]['BestRank'] += 1

                    #if auc_res == 1: #отключено AUC
                        #alg_A.meta[func]['AucRank'] += 1
                        
    
    def agregate_ranks(self):
        self.ranks_calc()
        for func in self.algs[0].meta:
            ranked_algs = [alg for alg in self.algs]
            ranked_algs.sort(key=lambda x: (x.meta[func]['BestRank'], 
                                            x.meta[func]['BestAvg']), reverse=True)
                                            #x.meta[func]['AucRank']), #отключено AUC
                                            
                   
            groups = []
            for k, g in groupby(ranked_algs, key=lambda x: (x.meta[func]['BestRank'])): #x.meta[func]['AucRank']#отключено AUC
                groups.append(list(g))
            
            for rank, group in enumerate(reversed(groups)):
                for alg in group:
                    alg.borda += rank
             
            for alg in groups[0]:
                alg.first += 1
                        
            for alg in list(chain(*groups[:3])):
                alg.top_three += 1
                
    def dump_stats(self, output):

        data = '/{}__genalg_test/'.format(datetime.strftime(datetime.now(), '%d_%m_%Y__%H_%M_%S'))
        filename = output + data
        
        if not os.path.exists(os.path.dirname(filename)):
            dir_name = os.path.dirname(filename)
            os.makedirs(dir_name)

        for func in self.test_funcs:
            fname = func.__name__.lower()
            file = filename + '/{}.csv'.format(fname)

            sort_key = lambda x: (x.meta[fname]['BestRank'], x.meta[fname]['AucRank'])
            self.algs.sort(key=sort_key, reverse=True)

            with open(file, "a") as f:
                f.write(';;;;BestRank;AucRank;BestAvg;R2Avg;RmseAvg;AucAvg;Catch;Steps;TimeAvg;fmin;r2_max;rmse_min;combs\n')
                for alg in self.algs:
                    out = alg.meta[fname]
                    f.write('{};{BestRank};{AucRank};{BestAvg};{R2Avg};{RmseAvg};{AucAvg};{Catch};{Steps};{TimeAvg};{fmin};{R2max};{RMSEmin};{comb}\n'.format(alg, **out))
                    
        self.algs.sort(key=lambda x: x.borda, reverse=True)
        file = filename + '/Agregated_stats.csv'

        with open(file, 'w') as f:
            f.write(';;;;Borda;First;TopThree\n')
            for alg in self.algs:
                f.write('{};{};{};{}\n'.format(alg, alg.borda, alg.first, alg.top_three))

               