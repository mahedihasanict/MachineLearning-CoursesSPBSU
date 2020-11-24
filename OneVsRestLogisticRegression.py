# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 21:38:40 2016

@author: Mahedi Hasan
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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


class OneVsRest():
    def train(self, X, y, classes=None):
        if classes is None:
            classes = np.unique(y)
        self.models = dict()
        self.classes = classes

        
        for c in classes:
            bin_c = y == c
            self.models[c] = LogisticRegression()
            self.models[c].fit(X, bin_c)
            
    def predict(self, X):
        scores = np.empty((len(self.classes), len(X)))
        for i, c in enumerate(self.classes):
            scores[i] = self.models[c].predict_proba(X)[:, 1]
        ic = np.argmax(scores, axis=0)
        return ic
        
        
n_splits = 20
algos = [OneVsRest()]
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

        
plt.figure(figsize=(20, 10))
labels = ('One vs rest', 'All pairs')
for i in range(len(algos)):
    sns.distplot(accuracy[i], label=labels[i])
plt.legend(loc='best')
plt.xlabel('Accuracy')
plt.show()