import numpy as np
from scipy.sparse.linalg import eigs
from tqdm import tqdm

class KernelPCA:

    def __init__(self, kernel, p):
        '''
        kernel: the used kernel for kernel PCA.
        p: the number of retained PC's.
        '''

        self.kernel = kernel 
        self.p = p

    def center(self, Kxx):

        bar_Kxx = (np.eye(self.n)-(1.0/self.n)*np.ones((self.n,self.n)))@self.Kxx@(np.eye(self.n)-(1.0/self.n)*np.ones((self.n,self.n)))
        return bar_Kxx

    def transform(self, X):

        features = []

        for x in tqdm(list(X)):

            x = x.reshape(-1,1)

            Kxx = self.kernel.eval_func(x, x)
            bar_Kxx = self.center(Kxx)

            E, U = eigs((1.0/32)*Kxx@Kxx.T, self.p+1)
            U /= U[1:]/np.sqrt(E[1:])

            features.append(np.expand_dims(U, 0))

        features = np.vstack(features)

        return features