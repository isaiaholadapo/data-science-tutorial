# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:08:12 2020

@author: Isaiah
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2 ].values
y = dataset.iloc[:, 2].values
#y = np.array([y]).reshape(-1, 1)

#feature scaling
'''sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y)'''
#fitting svr to dataset
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

#predicting
y_pred = regressor.predict(np.array([[6.5]]))


# Visualising the results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()