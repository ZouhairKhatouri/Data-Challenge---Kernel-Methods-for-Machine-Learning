from PatchExtractor import *

class SIFTExtractor:

    def __init__(self):

        self.pe = PatchesExtractor(s=3, verbose=False)
        
        self.filter = np.array([
                                [[0,1,0],[0,0,0],[0,-1,0]],\
                                [[0,0,1],[0,0,0],[-1,0,0]],\
                                [[0,0,0],[-1,0,1],[0,0,0]],\
                                [[-1,0,0],[0,0,0],[0,0,1]],\
                                [[0,-1,0],[0,0,0],[0,1,0]],\
                                [[0,0,-1],[0,0,0],[1,0,0]],\
                                [[0,0,0],[1,0,-1],[0,0,0]],\
                                [[1,0,0],[0,0,0],[0,0,-1]]
                               ])

    def pad(self, X):

        X = list(X)
        X_ = []
        for x in X:
            x = np.hstack([np.zeros(32)[:,None], x, np.zeros(32)[:,None]])
            x = np.vstack([np.zeros(34)[None,:], x, np.zeros(34)[None,:]])
            X_.append(np.expand_dims(x, 0))
        X = np.vstack(X_)

        return X

    def extract(self, X):

        assert X.shape[1] == 32 and X.shape[2] == 32
        X = X.mean(axis=-1)
        X = self.pad(X)
        assert X.shape[1] == 34 and X.shape[2] == 34

        patches = self.pe.extract(np.expand_dims(X, -1))

        SIFT = []
        for batch in patches:
               
            batch=batch.squeeze()
            batch=(batch[:,None,:,:,:,:]*self.filter[None,:,:,:,None,None]).sum(axis=2).sum(axis=2)
            batch=batch.argmax(axis=1)
            batch=batch.reshape(-1,8,4,8,4)
            batch=np.take(np.eye(8), batch, 0)
            batch=batch.transpose(0, 2, 4, 5, 1, 3)
            batch = batch.mean(axis=-1).mean(axis=-1)
            batch = batch.reshape(-1, 128)

            SIFT.append(batch)

        SIFT = np.vstack(SIFT)

        return SIFT


