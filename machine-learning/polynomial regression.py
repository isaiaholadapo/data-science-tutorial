# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 12:38:54 2020

@author: Isaiah
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, :-1]

