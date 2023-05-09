# -*- coding: utf-8 -*-
"""
Created on Mon May  2 17:41:49 2022

@author: ing06
"""

import numpy as np
import pandas as pd

# 인구밀도(X)와 절도발생률(Y)간의 관계
X = [ 59, 49, 75, 54, 78, 56, 60, 82, 69, 83, 88, 94, 47, 65, 89, 70]
Y = [ 209, 180, 195, 192, 215, 197, 208, 189, 213, 201, 214, 212, 205, 186, 200, 204]

X = np.reshape(X, (-1, 1))
Y = np.reshape(Y, (-1, 1))


# 회귀분석식 구하기
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 훈련용/테스트 데이터 분리
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

lr = LinearRegression()
lr.fit(X, Y)

print(lr.coef_[0])
print(lr.intercept_)

# 따라서 회귀식은 Y = 0.26X+182

# 데이터 분포를 산점도로 나타내기
import matplotlib.pyplot as plt 
plt.scatter(X, Y) 
plt.show()

# 결정계수(R-squared) 구하기
Y_predict = lr.predict(X)

print('R^2(Variance score) : {0:.3f}'.format(r2_score(Y, Y_predict)))

# X값이 58일 때, y값 예측
data_58 = {'x' : [58],
              'y' : []}
predict_58 = lr.predict(X = pd.DataFrame(data_58["x"]))
print("x값이 58일 때, y값 예측", predict_58)

