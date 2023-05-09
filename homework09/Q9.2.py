# -*- coding: utf-8 -*-
"""
Created on Mon May  2 18:12:45 2022

@author: ing06
"""
from sklearn import linear_model
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

diabetes_data = datasets.load_diabetes()
X = pd.DataFrame(diabetes_data.data)
y = diabetes_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

linear_regression = linear_model.LinearRegression()
linear_regression.fit(X_train, y_train)
prediction = linear_regression.predict(X_test)
print('a value = ', linear_regression.intercept_)
print('b balue =', linear_regression.coef_)

# 잔차: 실제 y값 - 예측 y값
residuals = y_test-prediction
SSE = (residuals**2).sum(); SST = ((y-y.mean())**2).sum()
R_squared = 1 - (SSE/SST)
print('R_squared = ', R_squared)
print('score = ', linear_regression.score(X_test, y_test))
print('Mean_Squared_Error = ', mean_squared_error(prediction, y_test))
print('RMSE = ', mean_squared_error(prediction, y_test)**0.5)
