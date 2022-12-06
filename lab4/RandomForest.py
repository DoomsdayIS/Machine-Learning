import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import random
from pprint import pprint
from sklearn import datasets
import time

from collections import Counter

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier


class RandomForest(BaseEstimator):
    def __init__(self, n_trees=10, max_depth=50, min_samples_split=2,
                 criterion='entropy', max_leaf_nodes=25):
        self.trees = []
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_leaf_nodes = max_leaf_nodes

    def _bootstrap(self, X, y):
        n_samples = len(X)
        indexes = np.random.choice(n_samples, n_samples, replace=True)
        return X.iloc[indexes, :], y.iloc[indexes]

    def most_common(self, y):
        return Counter(y).most_common(1)[0][0]

    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_trees):
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          min_samples_split=self.min_samples_split,
                                          criterion=self.criterion,
                                          max_leaf_nodes=self.max_leaf_nodes,
                                          max_features='sqrt')
            _x, _y = self._bootstrap(X, y)
            tree.fit(_x, _y)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        predictions = np.swapaxes(predictions, 0, 1)
        y_predicion = [self.most_common(tree_pred) for tree_pred in predictions]
        return np.array(y_predicion)
