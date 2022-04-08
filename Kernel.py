import numpy as np
from scipy.spatial.distance import cdist

class Kernel():

    def __init__(self, eval_func=None, metric=None, name=""):
        
        if metric is None:
            self.eval_func=eval_func
        elif eval_func is None and metric is not None:
            def eval_func(X, y):
                return cdist(X, y, metric=metric)
            self.eval_func = eval_func
            
        self.name = name
        
    def __add__(self, other):
        
        def new_eval(X, y):
            return self.eval_func(X, y) + other.eval_func(X, y)
        
        return Kernel(eval_func=new_eval, name=self.name+"+"+other.name)
    
    def __mul__(self, other):
        
        def new_eval(X, y):
            return self.eval_func(X, y)*other.eval_func(X, y)
        
        return Kernel(eval_func=new_eval, name=self.name+"x"+other.name)
    
    def __pow__(self, n):
        
        def new_eval(X, y):
            return self.eval_func(X, y)**n
        
        return Kernel(eval_func=new_eval, name=self.name+"**"+str(n))
    
    def __or__(self, other):
        
        def new_eval(X, y):
            return np.concatenate((self.eval_func(X, y)[:,:,None],other.eval_func(X, y)[:,:,None]), axis=2).min(axis=2)
        
        return Kernel(eval_func=new_eval, name="min("+self.name+","+other.name+")")
    
    def exp(kernel: '__main__.Kernel') -> '__main__.Kernel':
        
        def new_eval(X, y):
            return np.exp(kernel.eval_func(X, y))
        
        return Kernel(eval_func=new_eval, name="exp("+kernel.name+")")
    
    def rbf_exp(kernel: '__main__.Kernel', sigma:float=0.1) -> '__main__.Kernel':
        
        def new_eval(X, y):
            Kxx = np.vstack([kernel.eval_func(x.reshape(1,-1), x.reshape(1,-1)) for x in X])
            Kyy = np.hstack([kernel.eval_func(x.reshape(1,-1), x.reshape(1,-1)) for x in y])
            Kxy = kernel.eval_func(X, y)
            return np.exp(-(Kxx+Kyy-2*Kxy)/(2*sigma**2))
        
        return Kernel(eval_func=new_eval, name="rbf_exp("+kernel.name+")")

        
    def conv(self, W, m1=32, m2=32, ch=3):
        
        s = W.shape[0]
        P = np.arange(s)[None,None,:] + m2*np.arange(s)[None,:,None] + ch*m2*np.arange(ch)[:,None,None]
        T = np.arange(m1-s+1)[None,:] + m2*np.arange(m2-s+1)[:,None]
        I = T[:,:,None,None,None]+P[None,None,:,:,:]
        
        def new_eval(X, Y):
            
            X = X.reshape(-1,m1,m2,ch)
            Y = Y.reshape(-1,m1,m2,ch)
            
            WPX = []
            for x in X:
                Px = np.take(x.flatten(), I, 0) 
                WPx = (W[None,None,None,:,:]*Px).sum(axis=0).sum(axis=0).sum(axis=0)
                WPX.append(WPx.flatten())
            WPX = np.vstack(WPX)
            
            WPY = []
            for y in Y:
                Py = np.take(y.flatten(), I, 0) 
                WPy = (W[None,None,None,:,:]*Py).sum(axis=0).sum(axis=0).sum(axis=0)
                WPY.append(WPy.flatten())
            WPY = np.vstack(WPY)
            
            return self.eval_func(WPX, WPY)
        
        return Kernel(eval_func=new_eval, name="W*"+self.name)

def sum_min(x, y):
    return np.concatenate((x[:,None],y[:,None]),axis=1).sum(axis=-1).min(axis=-1)

def min_min(x, y):
    return np.concatenate((x[:,None],y[:,None]),axis=1).min(axis=-1).min(axis=-1)

def min_sum(x, y):
    return np.concatenate((x[:,None],y[:,None]),axis=1).min(axis=-1).sum(axis=-1)

def sum_sum(x, y):
    return np.concatenate((x[:,None],y[:,None]),axis=1).sum(axis=-1).sum(axis=-1)

def linear(x, y):
    return (x*y).sum()

def rbf(x, y, sigma=0.1):
    return np.exp(-(np.concatenate((x[:,None],-y[:,None]),axis=1).sum(axis=-1)**2).sum(axis=-1)/(2*sigma**2))

def laplace(x, y, sigma=0.1):
    return np.exp(-np.abs(np.concatenate((x[:,None],-y[:,None]),axis=1).sum(axis=-1)).sum(axis=-1)/sigma)

def cauchy(x, y):
    return (1.0/(1+(np.concatenate((x[:,None],-y[:,None]),axis=1).sum(axis=-1)**2))).sum(axis=-1)

SUM_MIN = Kernel(metric=sum_min, name="sum_min")
MIN_MIN = Kernel(metric=min_min, name="min_min")
MIN_SUM = Kernel(metric=min_sum, name="min_sum")
SUM_SUM = Kernel(metric=sum_sum, name="sum_sum")
LINEAR = Kernel(metric=linear, name="linear")
RBF = Kernel(metric=rbf, name="rbf")
LAPLACE = Kernel(metric=laplace, name="laplace")
CAUCHY = Kernel(metric=cauchy, name="cauchy")