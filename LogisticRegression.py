# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 01:52:22 2016

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


class OneVsRest(object):
    def train(self, X, y):
        self.classes = y
        self.models = LogisticRegression()
        self.models.fit(X, self.classes)
            
    def predict(self, X):
        scores = self.models.predict(X)
        return scores
        
        
n_splits = 20
algos = [OneVsRest()]
accuracy = np.empty((2, n_splits))
ss = ShuffleSplit(len(data_X), n_iter=n_splits, test_size=0.5)
for i, (train_ids, test_ids) in enumerate(ss):
    for j, a in enumerate(algos):
        a.train(data_X[train_ids], data_y[train_ids])
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