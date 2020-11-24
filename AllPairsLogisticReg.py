# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:17:02 2016

@author: Mahedi Hasan
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from itertools import combinations


class AllPairs():
    
    def train(self, X, y, classes=None):
        if classes is None:
            classes = np.unique(y)
        self.classes = classes
        self.models = dict()
        
        # Precompute indicator arrays for each class:
        self.ind = dict()
        for c in classes:
            self.ind[c] = np.equal(y, c)

        # Train a separate model for each pair of classes:
        self.comb = list(combinations(classes, 2))
        for c1, c2 in self.comb:
            self.models[(c1, c2)] = LogisticRegression()
            subset = np.logical_or(self.ind[c1], self.ind[c2])
            self.models[(c1, c2)].fit(X[subset], y[subset])
            
    def predict(self, X):
        n_classes = len(self.classes)
        n_instances = len(X)
        scores = np.zeros((n_classes, n_instances))
        for i, (c1, c2) in enumerate(self.comb):
            scores[[c1, c2]] += self.models[(c1, c2)].predict_proba(X).T
        ic = np.argmax(scores, axis=0)
        return ic