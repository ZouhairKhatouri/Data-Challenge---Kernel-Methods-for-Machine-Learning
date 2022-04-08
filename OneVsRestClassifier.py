import numpy as np
from copy import deepcopy

class OneVsRestClassifier:

    def __init__(self, clf, C=10) -> None:
        
        assert hasattr(clf, 'predict_proba')

        self.clfs = [deepcopy(clf) for _ in range(C)]
        self.C = C

    def fit(self, X, y):

        for i in range(self.C):
            y_i = (y==i).astype(int)
            self.clfs[i].fit(X, y_i)
    
    def predict(self, X):

        probas = []
        for i in range(self.C):
            pred = self.clfs[i].predict_proba(X)
            assert pred.shape[1] == 1
            probas.append(pred)
        
        probas = np.hstack(probas)
        pred = probas.argmax(axis=1)

        return probas