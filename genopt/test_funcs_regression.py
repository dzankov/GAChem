from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston, load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from genopt.genom import Categorial, Discrete, Continues
from hyperopt import hp

N_SPLITS = 3
RANDOM_STATE = 5

param_grid_genopt_mlp = {'hidden_layer_sizes': Categorial([(i,) for i in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]]),
                         'alpha': Categorial([10 ** i for i in range(-4, 5)]),
                         'activation': Categorial(['logistic', 'tanh', 'relu']),
                         'solver': Categorial(['lbfgs', 'adam'])
}

param_grid_genopt_svr = {'kernel':Categorial(['linear', 'poly', 'rbf', 'sigmoid']),
                         'C': Categorial([2 ** i for i in range(-5, 10)]),
                         'gamma':Categorial([2 ** i for i in range(-10, 4)])
}

param_grid_genopt_rf = {'bootstrap': Categorial([True, False]),
                        'max_depth': Categorial([10, 20, 30, 40, 50]),
                        'max_features': Categorial(['auto', 'sqrt']),
                        'min_samples_leaf': Categorial([1, 2, 4]),
                        'min_samples_split': Categorial([2, 5, 10]),
                        'n_estimators': Categorial([10, 20, 30, 40, 50])
}

param_grid_genopt_knn = {'n_neighbors': Categorial([i for i in range(1, 21)]),
                         'leaf_size':Categorial([1,2,3,5]),
                         'weights':Categorial(['uniform', 'distance']),
                         'algorithm':Categorial(['auto', 'ball_tree','kd_tree','brute']),
                         'metric': Categorial(['manhattan','euclidean','chebyshev'])
}

param_grid_genopt_kr = {'alpha':Categorial([10 ** i for i in range(-6, 0)]),
                  'gamma':Categorial([10 ** i for i in range(-12, 0)]),
                  'kernel':Categorial(['laplacian', 'rbf', 'linear', 'poly', 'sigmoid'])
}


param_grid_hyperopt_mlp = {'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(2 ** i,) for i in range(12)]),
                           'alpha': hp.choice('alpha', [10 ** i for i in range(-4, 5)]),
                           'activation': hp.choice('activation', ['logistic', 'tanh', 'relu']),
                           'solver': hp.choice('solver', ['lbfgs', 'adam'])
}

param_grid_hyperopt_svr = {'kernel':hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
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

param_grid_hyperopt_kr = {'alpha':hp.choice('alpha', [10 ** i for i in range(-6, 0)]),
                           'gamma':hp.choice('gamma', [10 ** i for i in range(-12, 0)]),
                           'kernel':hp.choice('kernel', ['laplacian', 'rbf', 'linear', 'poly', 'sigmoid'])
}     

class Diabetes:
    def __init__(self):
        data = load_diabetes()
        self.X = data.data
        self.y = data.target
        
        self.X_train, self.X_test, self.y_train, self.y_test =  train_test_split(
        data.data, data.target, test_size=0.2, random_state=RANDOM_STATE)
        
        
class Boston:
    def __init__(self):
        data = load_boston()
        self.X = data.data
        self.y = data.target
        
        self.X_train, self.X_test, self.y_train, self.y_test =  train_test_split(
        data.data, data.target, test_size=0.2, random_state=RANDOM_STATE)
        
class California:
    def __init__(self):
        data = fetch_california_housing()
        self.X = data.data[:500]
        self.y = data.target[:500]
        
        self.X_train, self.X_test, self.y_train, self.y_test =  train_test_split(
        data.data, data.target, test_size=0.2, random_state=RANDOM_STATE)
        
class MLP:
    param_grid_genopt = param_grid_genopt_mlp
    param_grid_hyperopt = param_grid_hyperopt_mlp
    comb = 840
    
    def __init__(self):
        pass

    def evaluate(self, params):
        mlp = MLPRegressor(**params, random_state=RANDOM_STATE)
        kf = KFold(n_splits=N_SPLITS, random_state=RANDOM_STATE)
        score = cross_val_score(mlp, self.X_train, self.y_train, cv=kf, scoring=make_scorer(mean_squared_error)).mean()
        return score
    
    def external_test(self, params):
        mlp = MLPRegressor(**params, random_state=RANDOM_STATE)
        mlp.fit(self.X_train, self.y_train)
        
        r2 = r2_score(self.y_test, mlp.predict(self.X_test))
        rmse = pow(mean_squared_error(self.y_test, mlp.predict(self.X_test)), 0.5)
        
        return r2, rmse
    
        

class SVRegressor:
    param_grid_genopt = param_grid_genopt_svr
    param_grid_hyperopt = param_grid_hyperopt_svr
    comb = 840
    
    def __init__(self):
        pass
    
    def evaluate(self, params):
        svr = SVR(**params)
        kf = KFold(n_splits=N_SPLITS, random_state=RANDOM_STATE)
        score = cross_val_score(svr, self.X_train, self.y_train, cv=kf, scoring=make_scorer(mean_squared_error)).mean()
        return score
    
    def external_test(self, params):
        svr = SVR(**params)
        svr.fit(self.X_train, self.y_train)
        
        r2 = r2_score(self.y_test, svr.predict(self.X_test))
        rmse = pow(mean_squared_error(self.y_test, svr.predict(self.X_test)), 0.5)
        
        return r2, rmse
    
class RFRegressor:
    param_grid_genopt = param_grid_genopt_rf
    param_grid_hyperopt = param_grid_hyperopt_rf
    comb = 900
    
    def __init__(self):
        pass
    
    def evaluate(self, params):
        rf = RandomForestRegressor(**params, random_state=RANDOM_STATE)
        kf = KFold(n_splits=N_SPLITS, random_state=RANDOM_STATE)
        score = cross_val_score(rf, self.X_train, self.y_train, cv=kf, scoring=make_scorer(mean_squared_error)).mean()
        return score
    
    def external_test(self, params):
        rf = RandomForestRegressor(**params, random_state=RANDOM_STATE)
        rf.fit(self.X_train, self.y_train)
        
        r2 = r2_score(self.y_test, rf.predict(self.X_test))
        rmse = pow(mean_squared_error(self.y_test, rf.predict(self.X_test)), 0.5)
        
        return r2, rmse
    
    
    
class KNNRegressor:
    param_grid_genopt = param_grid_genopt_knn
    param_grid_hyperopt = param_grid_hyperopt_knn
    comb = 1920
    
    def __init__(self):
        pass
    
    def evaluate(self, params):
        knn = KNeighborsRegressor(**params)
        kf = KFold(n_splits=N_SPLITS, random_state=RANDOM_STATE)
        score = cross_val_score(knn, self.X_train, self.y_train, cv=kf, scoring=make_scorer(mean_squared_error)).mean()
        return score
    
    def external_test(self, params):
        knn = KNeighborsRegressor(**params)
        knn.fit(self.X_train, self.y_train)
        
        r2 = r2_score(self.y_test, knn.predict(self.X_test))
        rmse = pow(mean_squared_error(self.y_test, knn.predict(self.X_test)), 0.5)
        
        return r2, rmse
    
class KRRegressor:
    param_grid_genopt = param_grid_genopt_kr
    param_grid_hyperopt = param_grid_hyperopt_kr
    comb = 360
    
    def __init__(self):
        pass
    
    def evaluate(self, params):
        krr = KernelRidge(**params)
        kf = KFold(n_splits=N_SPLITS, random_state=RANDOM_STATE)
        score = cross_val_score(krr, self.X_train, self.y_train, cv=kf, scoring=make_scorer(mean_squared_error)).mean()
        return score
    
    def external_test(self, params):
        krr = KernelRidge(**params)
        krr.fit(self.X_train, self.y_train)
        
        r2 = r2_score(self.y_test, krr.predict(self.X_test))
        rmse = pow(mean_squared_error(self.y_test, krr.predict(self.X_test)), 0.5)
        
        return r2, rmse
		
		
class DiabetesMLP(Diabetes, MLP):
    def __init__(self):
        super().__init__()
        self.min_loc = {'activation': 'relu', 'alpha': 10, 
                        'hidden_layer_sizes': (32,), 'solver': 'lbfgs'}
        self.fmin = 3025.426063016044
        self.r2 = 0.5074775295307301
        self.rmse = 55.72855730432512
        

class DiabetesSVR(Diabetes, SVRegressor):
    def __init__(self):
        super().__init__()
        self.min_loc = {'C': 64, 'gamma': 8, 'kernel': 'rbf'}
        self.fmin = 3037.949536206783
        self.r2 = 0.5385282622171976
        self.rmse = 53.94327749040544
        
        

class DiabetesRF(Diabetes, RFRegressor):
    def __init__(self):
        super().__init__()
        self.min_loc = {'bootstrap': True, 'max_depth': 20, 'max_features': 'sqrt', 
                        'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 40}
        self.fmin = 3403.069712319697
        self.r2 = 0.5277478135477158
        self.rmse = 54.56972488514209
        
        
class DiabetesKNN(Diabetes, KNNRegressor):
    def __init__(self):
        super().__init__()
        self.min_loc = {'algorithm': 'auto', 'leaf_size': 3, 'metric': 'chebyshev', 
                        'n_neighbors': 13, 'weights': 'distance'}
        self.fmin = 3648.910169120336
        self.r2 = 0.5307901276701548
        self.rmse = 54.393668010959374
        
class DiabetesKRR(Diabetes, KRRegressor):
    def __init__(self):
        super().__init__()
        self.min_loc = {'alpha': 0.1, 'gamma': 0.1, 'kernel': 'laplacian'}
        self.fmin = 3081.863984685884
        self.r2 = 0.5400347511757282
        self.rmse = 53.85515575212619
   

class BostonMLP(Boston, MLP):
    def __init__(self):
        super().__init__()
        self.min_loc = {'activation': 'logistic', 'alpha': 100, 
                        'hidden_layer_sizes': (1024,), 'solver': 'adam'}
        self.fmin = 36.477212745614565
        self.r2 = 0.2399229554246528
        self.rmse = 7.714231804474664
    

class BostonRF(Boston, RFRegressor):
    def __init__(self):
        super().__init__()
        self.min_loc = {'bootstrap': True, 'max_depth': 20, 'max_features': 'sqrt', 
                        'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 10}
        self.fmin = 16.192722566847177
        self.r2 = 0.8678671804508108
        self.rmse = 3.2163938852881597
 

class BostonKNN(Boston, KNNRegressor):
    def __init__(self):
        super().__init__()
        self.min_loc = {'algorithm': 'auto', 'leaf_size': 1, 'metric': 'manhattan', 
                        'n_neighbors': 20, 'weights': 'distance'}
        self.fmin = 42.26153169204048
        self.r2 = 0.5878530092755316
        self.rmse = 5.680544197405053

        
class BostonKRR(Boston, KRRegressor):
    def __init__(self):
        super().__init__()
        self.min_loc = {'alpha': 0.01, 'gamma': 0.0001, 'kernel': 'laplacian'}
        self.fmin = 22.08419496252978
        self.r2 = 0.8272983395839314
        self.rmse = 3.677156663189367
        

class CaliforniaMLP(California, MLP):
    def __init__(self):
        super().__init__()
        self.min_loc = {'activation': 'logistic', 'alpha': 1, 'hidden_layer_sizes': (4,), 'solver': 'lbfgs'}
        self.fmin = 0.9195262140359147
        self.r2 = -0.0004663414628454099
        self.rmse = 1.174873374937619
        

class CaliforniaRF(California, RFRegressor):
    def __init__(self):
        super().__init__()
        self.min_loc = {'bootstrap': True, 'max_depth': 10, 'max_features': 'auto', 
                        'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 30}
        self.fmin = 0.30293960276442794
        self.r2 = 0.7937607836324668
        self.rmse = 0.533427554516270
        
        
class CaliforniaKNN(California, KNNRegressor):
    def __init__(self):
        super().__init__()
        self.min_loc = {'algorithm': 'auto', 'leaf_size': 1, 'metric': 'manhattan', 
                        'n_neighbors': 7, 'weights': 'distance'}
        self.fmin = 1.0050649207327291
        self.r2 = 0.2964842591441532
        self.rmse = 0.9852052790371535
		
		
test_funcs = [DiabetesMLP, 
              DiabetesSVR, 
              DiabetesRF, 
              DiabetesKNN, 
              DiabetesKRR, 
              BostonMLP, 
              BostonRF, 
              BostonKNN, 
              BostonKRR,
              CaliforniaMLP,  
              CaliforniaRF, 
              CaliforniaKNN]