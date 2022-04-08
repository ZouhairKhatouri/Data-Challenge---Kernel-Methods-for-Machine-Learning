import numpy as np
import numpy.linalg as LA

class KernelPCA:

    def __init__(self, kernel, p):
        '''
        kernel: the used kernel for kernel PCA.
        p: the number of retained PC's.
        '''

        self.kernel = kernel 
        self.p = p

        # The fitted data shape:
        self.n = None

        # The Gramm matrix of the fitted data:
        self.Kxx = None

    def center(self):
        self.Kxx = (np.eye(self.n)-(1.0/self.n)*np.ones((self.n,self.n)))@self.Kxx@(np.eye(self.n)-(1.0/self.n)*np.ones((self.n,self.n)))

    def fit(self, X):

        self.n = X.shape[0]
        self.Kxx = self.kernel.eval_func(X, X)
        self.center()

        self.U, self.eigs, _ = LA.svd((1/self.n)*self.Kxx)
        self.U = self.U[:,:self.p]
        self.eigs = self.eigs[:self.p]
        self.U /= np.sqrt(self.eigs[None,:])