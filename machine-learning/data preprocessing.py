# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#importing library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

#label encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncode_X= LabelEncoder()
X[:, 0] = labelEncode_X.fit_transform(X[:, 0])
oneHotEncoder = OneHotEncoder(categorical_features= [0])
X =  oneHotEncoder.fit_transform(X).toarray()

labelEncode_y= LabelEncoder()
y = labelEncode_y.fit_transform(y)

# splitting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)