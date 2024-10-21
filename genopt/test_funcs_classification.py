from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_breast_cancer
from hyperopt import hp

N_SPLITS = 3
RANDOM_STATE = 5

param_grid_genopt_mlp = {'hidden_layer_sizes': [(i,) for i in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]],
                         'alpha': [10 ** i for i in range(-4, 5)],
                         'activation': ['logistic', 'tanh', 'relu'],
                         'solver': ['lbfgs', 'adam']
}

param_grid_genopt_svc = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
                         'C': [2 ** i for i in range(-5, 10)],
                         'gamma':[2 ** i for i in range(-10, 4)]
}

param_grid_genopt_rf = {'bootstrap': [True, False],
                        'max_depth': [10, 20, 30, 40, 50],
                        'max_features': ['auto', 'sqrt'],
                        'min_samples_leaf': [1, 2, 4],
                        'min_samples_split': [2, 5, 10],
                        'n_estimators': [10, 20, 30, 40, 50]
}

param_grid_genopt_knn = {'n_neighbors': [i for i in range(1, 21)],
                         'leaf_size':[1,2,3,5],
                         'weights':['uniform', 'distance'],
                         'algorithm':['auto', 'ball_tree','kd_tree','brute'],
                         'metric': ['manhattan','euclidean','chebyshev']
}


param_grid_hyperopt_mlp = {'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(2 ** i,) for i in range(12)]),
                           'alpha': hp.choice('alpha', [10 ** i for i in range(-4, 5)]),
                           'activation': hp.choice('activation', ['logistic', 'tanh', 'relu']),
                           'solver': hp.choice('solver', ['lbfgs', 'adam'])
}

param_grid_hyperopt_svc = {'kernel':hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                           'C': hp.choice('C', [2 ** i for i in range(-5, 10)]),
                           'gamma':hp.choice('gamma', [2 ** i for i in range(-10, 4)])
}

param_grid_hyperopt_rf = {'bootstrap': hp.choice('bootstrap', [True, False]),
                          'max_depth': hp.choice('max_depth', [10, 20, 30, 40, 50]),
                          'max_features': hp.choice('max_features', ['auto', 'sqrt']),
                          'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4]),
                          'min_samples_split': hp.choice('min_samples_split', [2, 5, 10]),
                          'n_estimators': hp.choice('n_estimators', [10, 20, 30, 40, 50])
}

param_grid_hyperopt_knn = {'n_neighbors': hp.choice('n_neighbors', [i for i in range(1, 21)]),
                           'leaf_size':hp.choice('leaf_size', [1,2,3,5]),
                           'weights':hp.choice('weights', ['uniform', 'distance']),
                           'algorithm':hp.choice('algorithm', ['auto', 'ball_tree','kd_tree','brute']),
                           'metric': hp.choice('metric', ['manhattan','euclidean','chebyshev'])
}


class Cancer:
    def __init__(self):
        data = load_breast_cancer()
        self.X = data.data
        self.y = data.target
        
class SVClassifier:
    param_grid_genopt = param_grid_genopt_svc
    param_grid_hyperopt = param_grid_hyperopt_svc
    comb = 840
    
    def __init__(self):
        pass
    
    def evaluate(self, params):
        svc = SVC(**params)
        kf = KFold(n_splits=N_SPLITS, random_state=RANDOM_STATE)
        score = cross_val_score(svc, self.X, self.y, cv=kf).mean()
        return -score
    
class RFClassifier:
    param_grid_genopt = param_grid_genopt_rf
    param_grid_hyperopt = param_grid_hyperopt_rf
    comb = 900
    
    def __init__(self):
        pass
    
    def evaluate(self, params):
        rfc = RandomForestClassifier(**params)
        kf = KFold(n_splits=N_SPLITS, random_state=RANDOM_STATE)
        score = cross_val_score(rfc, self.X, self.y, cv=kf).mean()
        return -score
    
class MLP:
    param_grid_genopt = param_grid_genopt_mlp
    param_grid_hyperopt = param_grid_hyperopt_mlp
    comb = 840
    
    def __init__(self):
        pass
    
    def evaluate(self, params):
        mlp = MLPClassifier(**params)
        kf = KFold(n_splits=N_SPLITS, random_state=RANDOM_STATE)
        score = cross_val_score(mlp, self.X, self.y, cv=kf).mean()
        return -score
    
class KNNClasifier:
    param_grid_genopt = param_grid_genopt_knn
    param_grid_hyperopt = param_grid_hyperopt_knn
    comb = 1920
    
    def __init__(self):
        pass
    
    def evaluate(self, params):
        knn = KNeighborsClassifier(**params)
        kf = KFold(n_splits=N_SPLITS, random_state=RANDOM_STATE)
        score = cross_val_score(knn, self.X, self.y, cv=kf).mean()
        return -score
		
		
class CancerSVC(Cancer, SVClassifier):
    def __init__(self):
        super().__init__()
        self.min_loc = None
        self.fmin = None
        
class CancerRF(Cancer, RFClassifier):
    def __init__(self):
        super().__init__()
        self.min_loc = {'bootstrap': False, 'max_depth': 10, 'max_features': 'auto', 
                        'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 20}
        self.fmin = -0.964850615114235
        
class CancerMLP(Cancer, MLP):
    def __init__(self):
        super().__init__()
        self.min_loc = None
        self.fmin = None
        
class CancerKNN(Cancer, KNNClasifier):
    def __init__(self):
        super().__init__()
        self.min_loc = {'algorithm': 'auto', 'leaf_size': 1, 'metric': 'manhattan', 'n_neighbors': 5, 'weights': 'distance'}
        self.fmin = -0.931458699472759
		
		
test_funcs = [CancerSVC, 
              CancerRF, 
              CancerMLP, 
              CancerKNN]