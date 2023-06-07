"""
File for quick experiments
"""

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from scipy.spatial import distance

def customMetric(x, y, weights):
    weightIndex = np.where((weights[:, 0:-1] == y).all(axis=1))[0]
    dist = distance.euclidean(x, y)
    try:
        wIndex = weightIndex[0]
    except IndexError as iexc:
        wIndex = weightIndex
    dist = np.multiply(float(dist), weights[wIndex, -1])
    return float(dist)


datagenerator = np.random.default_rng()
data = datagenerator.random(size=(1000, 50))
labels = datagenerator.integers(0, 3, size=1000)
w = datagenerator.random(size=(1000, 1))
w = np.concatenate([data, w], axis=1)
basePath = './Data/dataset/'
train = np.load(basePath + 'training_unscaled.npy')
val = np.load(basePath + 'validation_unscaled.npy')
test = np.load(basePath + 'test_unscaled.npy')
w = datagenerator.random(size=(train.shape[0], 1))
w = np.concatenate([train[:, 0:-1], w], axis=1)
clf = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='pyfunc', metric_params={'func': customMetric, 'weights': w})
clf.fit(train[:, 0:-1], train[:, -1])
clf.predict(train[:, 0:-1])
