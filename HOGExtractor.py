import numpy as np

from PatchExtractor import *


class HOGExtractor:

    def __init__(self):

        self.pe = PatchesExtractor(s=3, verbose=False)

        self.filter = np.array([
            [[0, 1, 0], [0, 0, 0], [0, -1, 0]], \
            [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]
        ])

    def pad(self, X):

        X = list(X)
        X_ = []
        for x in X:
            x = np.hstack([np.zeros(32)[:, None], x, np.zeros(32)[:, None]])
            x = np.vstack([np.zeros(34)[None, :], x, np.zeros(34)[None, :]])
            X_.append(np.expand_dims(x, 0))
        X = np.vstack(X_)

        return X

    def extract(self, X):

        assert X.shape[1] == 32 and X.shape[2] == 32
        X = X.mean(axis=-1)
        X = self.pad(X)
        assert X.shape[1] == 34 and X.shape[2] == 34

        patches = self.pe.extract(np.expand_dims(X, -1))

        HOG = []
        for batch in patches:
            batch = batch.squeeze()
            batch = (batch[:, None, :, :, :, :] * self.filter[None, :, :, :, None, None]).sum(axis=2).sum(axis=2)
            Gx = batch[:,1,:,:]
            Gy = batch[:,0,:,:]
            batch = ((180.0/np.pi)*np.arctan(Gy/(Gx+1e-10))).astype(int)
            batch = np.take(np.eye(180), batch, 0)
            batch = batch.mean(axis=1).mean(axis=1)

            HOG.append(batch)

        HOG = np.vstack(HOG)

        return HOG