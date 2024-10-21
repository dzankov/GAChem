from operator import itemgetter
from hyperopt import fmin, tpe, rand, hp, STATUS_OK, Trials


class HyperOpt:
    
    def __init__(self, algo='tpe'):
    
        self.algo = algo
        self.fitness = None
        
    
    def initialize(self, param_grid):
    
        if self.algo == 'tpe':
             self.method = tpe.suggest
             
        elif self.algo == 'random':
             self.method = rand.suggest
        
        self.param_grid = param_grid
        
        return self
        
    
    def set_fitness(self, func):
        self.fitness = func
    
    
    def fit(self, n_iter=100):
        
        def objective(params):
            model = self.fitness()
            loss = model.evaluate(params)
            return {'loss': loss, 'params': params, 'status': STATUS_OK}
    
        trials = Trials()  
        results = fmin(
            objective, self.param_grid, algo=self.method,
            trials=trials, max_evals=n_iter)
        
        self.results = trials.results
        
        return self
    
    def best_solution(self):
        best = min(self.results, key=itemgetter('loss'))['params']
        return best
    
    
    def best_score(self):
        score = res = min(self.results, key=itemgetter('loss'))['loss']
        return score
    
    
    def loss_scores(self):
        scores = [n_iter['loss'] for n_iter in self.results]
        return scores