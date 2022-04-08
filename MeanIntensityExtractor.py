from PatchExtractor import *

class MeanIntensityExctractor:

    def __init__(self):

        self.pe = PatchesExtractor(s=3, verbose=False)

    def pad(self, X):

        X = list(X)
        X_ = []
        for x in X:
            x = np.hstack([np.repeat(np.zeros(32)[:,None,None],3,2), x, np.repeat(np.zeros(32)[:,None,None],3,2)])
            x = np.vstack([np.repeat(np.zeros(34)[None,:,None],3,2), x, np.repeat(np.zeros(34)[None,:,None],3,2)])
            X_.append(np.expand_dims(x, 0))
        X = np.vstack(X_)

        return X

    def extract(self, X):

        assert X.shape[1] == 32 and X.shape[2] == 32
        X = self.pad(X)
        assert X.shape[1] == 34 and X.shape[2] == 34

        patches = self.pe.extract(X)

        MEANS = []
        for batch in patches:
               
            batch=batch.squeeze()
            batch=batch.sum(axis=1).sum(axis=1)
            batch-=batch.min()
            batch/=batch.max()
            batch*=31
            batch=batch.astype(int)
            batch=np.take(np.eye(32),batch,0)
            batch=batch.transpose(0, 1, 4, 2, 3)
            batch=batch.mean(axis=-1).mean(axis=-1)
            batch=batch.reshape(-1, 3*32)

            MEANS.append(batch)

        MEANS = np.vstack(MEANS)

        return MEANS