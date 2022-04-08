# Imports:

from SIFTExtractor import *
from HOGExtractor import *
from Kernel import *
from SVC import *
from OneVsRestClassifier import *

import pandas as pd
import numpy as np

# Train set import:

Xtr = np.array(pd.read_csv('Xtr.csv',header=None,sep=',',usecols=range(3072)))
Ytr = np.array(pd.read_csv('Ytr.csv',sep=',',usecols=[1])).squeeze()

assert 10 == len(np.unique(Ytr))

# Features extraction:

se = SIFTExtractor()
he = HOGExtractor()

sift = se.extract(Xtr.reshape(-1, 32, 32, 3))
hog = he.extract(Xtr.reshape(-1, 32, 32, 3))

# Kernel definition:

sigma = 15.0
def rbf(x, y):
    return np.exp(-(np.concatenate((x[:,None],-y[:,None]),axis=1).sum(axis=-1)**2).sum(axis=-1)/(2*sigma**2))
RBF = Kernel(metric=rbf, name="rbf")

# Model training:

clf = OneVsRestClassifier(SVC(1.0, kernel=RBF), 10).fit(np.hstack([sift, hog]), Ytr)

# Test set import:

Xte = np.array(pd.read_csv('Xte.csv',header=None,sep=',',usecols=range(3072)))

# Features extraction:

sift = se.extract(Xte.reshape(-1, 32, 32, 3))
hog = he.extract(Xte.reshape(-1, 32, 32, 3))

# Model prediction:

Yte = clf.predict(np.hstack([sift, hog]))

# Predictions saving:

Yte = {'Prediction' : Yte}
dataframe = pd.DataFrame(Yte)
dataframe.index += 1
dataframe.to_csv('Yte_pred.csv',index_label='Id')