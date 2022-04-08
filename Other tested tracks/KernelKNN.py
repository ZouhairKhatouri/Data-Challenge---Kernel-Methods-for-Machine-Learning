import numpy as np

class KernelKNN:
    
    def __init__(self, kernel, k):
        
        self.kernel = kernel
        self.k = k
    
    def fit(self, X, y):
        
        self.X = X
        self.y = y
        
    def predict(self, u):
        
        self.Kxu = self.kernel.eval_func(u, self.X)
        return np.take(np.eye(10),\
                       np.take(self.y, np.argsort(self.Kxu, axis=-1)[:,-self.k:], axis=0),\
                       axis=0).sum(axis=1).argmax(axis=-1)