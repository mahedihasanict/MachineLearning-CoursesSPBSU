# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:57:39 2016

@author: Mahedi Hasan
"""

import numpy as np
import pandas as pd
from OneVsRestLogisticReg import OneVsRest
from AllPairsLogisticReg import AllPairs
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


    
n_splits = 20
algos = [OneVsRest(),AllPairs()]
accuracy = np.empty((2, n_splits))
ss = ShuffleSplit(len(data_X), n_iter=n_splits, test_size=0.5)
for i, (train_ids, test_ids) in enumerate(ss):
    for j, a in enumerate(algos):
        a.train(data_X[train_ids], data_y[train_ids],
                classes=np.unique(data_y))
        preds = a.predict(data_X[test_ids])
        accuracy[j, i] = np.equal(preds, data_y[test_ids]).mean()
        

print('One vs rest with LR: Mean={:.1%}, SD={:.1%}'.format(
        accuracy[0].mean(), accuracy[0].std()))
print('All pairs with LR: Mean={:.1%}, SD={:.1%}'.format(
        accuracy[1].mean(), accuracy[1].std()))

        
plt.figure(figsize=(20, 10))
labels = ('One vs rest with LR', 'All pairs with LR')
for i in range(len(algos)):
    sns.distplot(accuracy[i], label=labels[i])
plt.legend(loc='best')
plt.xlabel('Accuracy')
plt.show()