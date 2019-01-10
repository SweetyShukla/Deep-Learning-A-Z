# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 10:17:04 2019

@author: hp
"""
#import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset

dataset=pd.read_csv('Data.csv')

#extract dependent and independent values in different arrays
x=dataset.iloc[:,:-1].values

y=dataset.iloc[:,-1].values

#handling missing values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean",axis=0)
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])

#encoding categorical data

from sklearn.preprocessing import LabelEncoder
encoded=LabelEncoder()
encoded.fit_transform(x[:,0])
x[:,0]=encoded.fit_transform(x[:,0])

#dummy encoding
from sklearn.preprocessing import OneHotEncoder
one= OneHotEncoder(categorical_features=[0])
x=one.fit_transform(x).toarray()

from sklearn.preprocessing import LabelEncoder
label_y=LabelEncoder()
y=label_y.fit_transform(y)

#spliting dataset into training set and test set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=0) 

#feature scaling

from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
x_train= sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)





























