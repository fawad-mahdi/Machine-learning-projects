# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:18:16 2017

@author: Fawad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


cancer=pd.read_csv('Question2.csv', header= None)

X=cancer.iloc[:,2:].values
Y=cancer.iloc[:,1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)
             
Y_new=Y.reshape((len(Y),1))

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y_new, test_size = 0.20, random_state = 0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
print(classifier.intercept_)
coef= classifier.coef_
classifier.get_params

Sample_20= classifier.predict_proba(X[19])
#prob = classifier.predict_proba(pd.DataFrame({'balance': [19]}))

pred_y = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(pred_y, y_test))


from sklearn.metrics import classification_report  

print(classification_report(pred_y, y_test, digits=3))
