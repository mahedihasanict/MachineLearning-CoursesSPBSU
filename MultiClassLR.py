# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:54:44 2016

@author: Mahedi Hasan
"""


from sklearn.linear_model import LogisticRegression


class MCLR():
    def train(self, X, y):
        self.classes = y
        self.models = LogisticRegression()
        self.models.fit(X, self.classes)
            
    def predict(self, X):
        scores = self.models.predict(X)
        return scores