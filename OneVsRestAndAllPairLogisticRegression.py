# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 21:35:22 2016

@author: Mahedi Hasan
"""

#matplotlib inline

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from itertools import combinations
from sklearn.cross_validation import ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns



import_options = dict(sep=',',
    names=['att{}'.format(i) for i in range(1, 17)] + ['class'])
train_data = pd.read_table('F:\\Masters SPBSU\\3rd semester\\Machine learning\\ML data\\pendigits.tra',
                               **import_options)
test_data = pd.read_table('F:\\Masters SPBSU\\3rd semester\\Machine learning\\ML data\\pendigits.tes',
                               **import_options)
merged = pd.concat([train_data, test_data])
data_X = merged.loc[:,'att1':'att16'].values
data_y = merged.loc[:, 'class'].values
print('#Rows: {}'.format(len(merged)))
merged.head()

class OneVsRest(object):
    def train(self, X, y, classes=None):
        if classes is None:
            classes = np.unique(y)
        self.models = dict()
        self.classes = classes
        #print X
        #print y
        
        for c in classes:
            bin_c = y == c
            #print bin_c
            self.models[c] = LogisticRegression()
            self.models[c].fit(X, bin_c)
            
    def predict(self, X):
        scores = np.empty((len(self.classes), len(X)))
        for i, c in enumerate(self.classes):
            scores[i] = self.models[c].predict_proba(X)[:, 1]
        ic = np.argmax(scores, axis=0)
        return ic


class AllPairs(object):
    
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
        
n_splits = 20
algos = [OneVsRest(), AllPairs()]
accuracy = np.empty((2, n_splits))
ss = ShuffleSplit(len(data_X), n_iter=n_splits, test_size=0.5)
for i, (train_ids, test_ids) in enumerate(ss):
    for j, a in enumerate(algos):
        a.train(data_X[train_ids], data_y[train_ids],
                classes=np.unique(data_y))
        preds = a.predict(data_X[test_ids])
        accuracy[j, i] = np.equal(preds, data_y[test_ids]).mean()
        
        
print('One vs rest: Mean={:.1%}, SD={:.1%}'.format(
        accuracy[0].mean(), accuracy[0].std()))
print('All pairs: Mean={:.1%}, SD={:.1%}'.format(
        accuracy[1].mean(), accuracy[1].std()))
        
plt.figure(figsize=(20, 10))
labels = ('One vs rest', 'All pairs')
for i in range(len(algos)):
    sns.distplot(accuracy[i], label=labels[i])
plt.legend(loc='best')
plt.xlabel('Accuracy')
plt.show()
        
