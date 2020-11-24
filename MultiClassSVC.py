# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 15:05:32 2016

@author: Mahedi Hasan
"""


from sklearn.svm import SVC


class MCSVC():
    def train(self, X, y):
        self.classes = y
        self.models = SVC()
        self.models.fit(X, self.classes)
            
    def predict(self, X):
        scores = self.models.predict(X)
        return scores