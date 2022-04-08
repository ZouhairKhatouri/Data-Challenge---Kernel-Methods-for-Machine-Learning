import numpy as np
import scipy.optimize as optimize

class SVC:

    def __init__(self, C, kernel, epsilon = 1e-3):

        self.C = C
        self.kernel = kernel
        
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None
        self.X = None
        self.y = None

    def fit(self, X, y):

        #### You might define here any variable needed for the rest of the code
        N = len(y)
        K = self.kernel.eval_func(X, X)
        self.X = X
        self.y = y

        # Lagrange dual problem
        def loss(alpha):
            return 0.5*(y*alpha)[np.newaxis, :]@K@(y*alpha)[:, np.newaxis]-alpha.sum(axis=0)#'''--------------dual loss ------------------ '''

        # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            return y*(K@(y*alpha)[:, np.newaxis]).squeeze(1) - np.ones(N)#'''----------------partial derivative of the dual loss wrt alpha-----------------'''

        fun_eq = lambda alpha: (y*alpha).sum(axis=0) 
        jac_eq = lambda alpha: y 
        fun_ineq_1 = lambda alpha: alpha 
        jac_ineq_1 = lambda alpha: np.eye(N) 
        fun_ineq_2 = lambda alpha: (self.C - alpha) 
        jac_ineq_2 = lambda alpha: - np.eye(N) 
        constraints = ({'type': 'eq', 'fun': fun_eq, 'jac': jac_eq},{'type': 'ineq', 'fun': fun_ineq_1 , 'jac': jac_ineq_1},{'type': 'ineq', 'fun': fun_ineq_2 , 'jac': jac_ineq_2})
        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),x0=np.ones(N),method='SLSQP',jac=lambda alpha: grad_loss(alpha),constraints=constraints)
        
        self.alpha = optRes.x
        ## Assign the required attributes
        supportIndices = [i for i in range(N) if self.epsilon < self.alpha[i] and self.alpha[i] < (1 - self.epsilon)*self.C]
        self.support = X[supportIndices] 
        self.b = (y[supportIndices] - ((y*self.alpha)[np.newaxis, :]@self.kernel.eval_func(X,self.support)).squeeze(0)).mean(axis=0)

        self.norm_f = np.sqrt((self.alpha*y)[np.newaxis, :]@K@(self.alpha*y)[:, np.newaxis])

    ### Implementation of the separting function $f$
    def separating_function(self, x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        return (self.alpha*self.y).T@self.kernel.eval_func(self.X, x)

    def predict(self, X):
        """ Predict y values in {0, 1} """
        d = self.separating_function(X)
        return float(1.0/(1.0+np.exp(-(d+self.b))) > 0.5)

    def predict_proba(self, X):
        """ Predict y values in [0, 1] """
        d = self.separating_function(X)
        return 1.0/(1.0+np.exp(-(d+self.b)))