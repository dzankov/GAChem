import random
from math import exp


def choice(options):
    return options[random.randint(0, len(options) - 1)]


def uniform(low, high):
    return random.uniform(low, high)


def loguniform(low, high):
    return exp(uniform(low, high))


def quniform(low, high, q):
    return round(uniform(low, high) / q) * q


def qloguniform(low, high, q):
    return round(loguniform(low, high) / q) * q


class Categorial:
    
    def __init__(self, prange):
        self.prange = prange
        
    def get_point(self):
        return choice(self.prange)


class Integer:

    def __init__(self, low, high, q=1):
        self.prange = low, high
        self.q = q

    def get_point(self):
        return round(quniform(*self.prange, self.q))


class Discrete:

    def __init__(self, low, high, q=1, dist=None):
        self.prange = low, high
        self.q = q
        
        if dist == 'quniform':
            self.dist = quniform
        elif dist == 'qlog-uniform':
            self.dist = qloguniform
        else:
            assert 'Unknown distribution type'
            
    def get_point(self):
        return self.dist(*self.prange, self.q)
    

class Continues:

    def __init__(self, low, high, dist=None):
        self.prange = low, high
        
        if dist == 'uniform':
            self.dist = uniform
        elif dist == 'log-uniform':
            self.dist = loguniform
        else:
            assert 'Incorrect distribution type'
            
    def get_point(self):
        return self.dist(*self.prange)

    