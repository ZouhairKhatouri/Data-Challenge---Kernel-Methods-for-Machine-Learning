import numpy as np
from tqdm import tqdm

class CrossEntropyClassifier:

    def __init__(self, d, C):

        self.d = d
        self.C = C
        self.W = np.random.randn(self.d, self.C)

    def loss(self, X, y):

        logits = X@self.W
        return (\
            (-np.take_along_axis(logits, y, axis=1)+logits.max(axis=1))\
            -np.exp(-logits+logits.max(axis=1, keepdim=True)).sum(axis=1)\
                ).mean()

    def loss_grad(self, X, y):
        
        E = np.take(np.eye(self.C), y, axis=0)
        logits = X@self.W
        logits = -logits+logits.max(axis=1, keepdim=True)
        P = np.exp(logits)/logits.sum(axis=1, keepdim=True)

        return ((E-P)[:,None,:]*X[:,:,None]).mean(axis=0)

    def fit(self, X, y, N_itt=10, lr=1e-3):

        n, d = X.shape

        assert d==self.d, "Non-compatible shape."

        self.W = np.random.randn((self.d, self.C))

        for itt in tqdm(range(N_itt)):

            L = self.loss(X, y)
            gradL = self.loss_grad(X, y)
            self.W -= lr * gradL

            print(f"itt={itt+1}/{N_itt}, loss={L:.2f}")


    def predict(self, X):
        
        return (X@self.dW).argmax(axis=1)