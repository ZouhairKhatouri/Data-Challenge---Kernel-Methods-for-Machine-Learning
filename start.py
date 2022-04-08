# Imports:

from SIFTExtractor import *
from HOGExtractor import *
from MeanExtractor import *
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
mie = MeanIntensityExctractor()

sift = se.extract(Xtr.reshape(-1, 32, 32, 3))
hog = he.extract(Xtr.reshape(-1, 32, 32, 3))
means = mie.extract(Xtr.reshape(-1, 32, 32, 3))

# Kernel definition:

sigma = 15.0
def rbf(x, y, sigma=0.1):
    return np.exp(-(x-y)**2/(2*sigma**2)).sum()
RBF = Kernel(metric=rbf, name="rbf")

# Model training:

clf = OneVsRestClassifier(SVC(1.0, kernel=RBF), 10).fit(np.hstack([sift, hog, means]), Ytr)

# Test set import:

Xte = np.array(pd.read_csv('Xte.csv',header=None,sep=',',usecols=range(3072)))

# Features extraction:

sift = se.extract(Xte.reshape(-1, 32, 32, 3))
hog = he.extract(Xte.reshape(-1, 32, 32, 3))
means = mie.extract(Xte.reshape(-1, 32, 32, 3))

# Model prediction:

Yte = clf.predict(np.hstack([sift, hog, means]))

# Predictions saving:

Yte = {'Prediction' : Yte}
dataframe = pd.DataFrame(Yte)
dataframe.index += 1
dataframe.to_csv('Yte_pred.csv',index_label='Id')