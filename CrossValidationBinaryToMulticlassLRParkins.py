# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 16:04:35 2016

@author: Mahedi Hasan
"""

import numpy as np
import pandas as pd
from OneVsRestLogisticReg import OneVsRest
#from AllPairsLogisticReg import AllPairs
from sklearn.cross_validation import ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns


import_options = dict(sep=',',names=['att{}'.format(i) for i in range(1, 23)])
whole_data = pd.read_table('F:\\Masters SPBSU\\3rd semester\\Machine learning\\ML data\\parkinsons_updrs.data',
                               **import_options)
data_X = whole_data.loc[:,'att7':'att16'].values
data_y = whole_data.loc[:, 'att6'].values
data_y = np.array(data_y).astype(int)
print('#Rows: {}'.format(len(whole_data)))
whole_data.head()


    
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
        

print('One vs rest with LR: Mean={:.1%}, SD={:.1%}'.format(
        accuracy[0].mean(), accuracy[0].std()))


        
plt.figure(figsize=(10, 5))
labels = ('One vs rest with LR')
for i in range(len(algos)):
    print (accuracy[i])
    sns.distplot(accuracy[i], label='One vs rest with LR')
    #sns.distplot(accuracy[i], label=labels[i])
plt.legend(loc='best')
plt.xlabel('Accuracy')
plt.show()