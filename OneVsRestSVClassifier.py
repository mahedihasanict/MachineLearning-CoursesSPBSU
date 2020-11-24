# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:26:19 2016

@author: Mahedi Hasan
"""

import numpy as np
from sklearn.svm import SVC


class OneVsRest():
    def train(self, X, y, classes=None):
        if classes is None:
            classes = np.unique(y)
        self.models = dict()
        self.classes = classes

        
        for c in classes:
            bin_c = y == c
            self.models[c] =SVC(probability=True)
            self.models[c].fit(X, bin_c)
            
    def predict(self, X):
        scores = np.empty((len(self.classes), len(X)))
        for i, c in enumerate(self.classes):
            scores[i] = self.models[c].predict_proba(X)[:, 1]
        ic = np.argmax(scores, axis=0)
        return ic