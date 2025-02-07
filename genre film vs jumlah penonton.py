# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 06:49:05 2020

@author: Elvira
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import seaborn as sns
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")


#import Dataset ke Python
Data=pd.read_excel("UTS.xlsx")
X=Data[['X']]
Y=Data['Y']
print(Data)


#Hitung jumlah kolom dan baris pada Dataset
print("Total baris pada dataset= {}".format(Data.shape[0]))
print("Total kolom pada dataset= {}".format(Data.shape[1]))



X_train, X_test, Y_train, Y_test = train_test_split(X,Y,
                                                    test_size = 0.05,
                                                    random_state = 0)
X_with_constant = sm.add_constant(X_train)
model= sm.OLS(Y_train, X_with_constant)

results= model.fit()
print(results.params)
print(results.summary())

X_test= sm.add_constant(X_test)
print(X_test)
Y_pred= results.predict(X_test)
residual= Y_test - Y_pred
print(residual)

sns.distplot(residual)

fig, ax = plt.subplots(figsize=(6,2.5))
_,(__. ___, r) = sp.stats.probplot(residual, plot=ax, fit=True)
