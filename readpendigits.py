# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 04:04:09 2016

@author: Mahedi Hasan
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
#from itertools import combinations
from sklearn.cross_validation import ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import ndarray

import_options = dict(sep=',',names=['att{}'.format(i) for i in range(1, 23)])
whole_data = pd.read_table('F:\\Masters SPBSU\\3rd semester\\Machine learning\\ML data\\parkinsons_updrs.data', **import_options)
#test_data = pd.read_table('F:\\Masters SPBSU\\3rd semester\\Machine learning\\ML data\\pendigits.tes',
#                               **import_options)
                               
#print(train_data.Name)
#data_X = train_data.loc[:,'Jitter(%)':'PPE'].values
data_X = whole_data.loc[:,'att7':'att16'].values
data_y = whole_data.loc[:, 'att6'].values
data_y = np.array(data_y).astype(int)
print(data_y)