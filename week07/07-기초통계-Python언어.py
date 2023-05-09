# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:30:51 2022

@author: Park
"""

blood = ['A', 'A', 'A', 'B', 'B', 'AB', 'O']
import numpy as np
np.unique(blood, return_counts=True)

import pandas as pd
pd.Series(blood).value_counts()   # 각 값의 개수를 나타냄

import seaborn as sns
sns.countplot(blood)   # 빈도 그래프로 나타냄

x = [1, 1, 1, 2, 3, 5, 5, 7, 8, 9]
hist, edges = np.histogram(x, 4)
sns.distplot(x, bins=4, kde=False)

x = [100, 100, 200, 400, 500]
import numpy
numpy.mean(x)

y = [100, 100, 200, 400, 1700]
numpy.mean(y)
numpy.median(x)
numpy.median(y)

# 중앙값(median): 데이터가 짝수개일 경우 가운데 두 값의 평균
numpy.median([100, 200, 300, 400])

from scipy.stats import mode
mode(x)  # x의 최빈값

import numpy
x = [1, 1, 2, 3, 3, 3, 4, 5, 5, 7]
numpy.min(x)
numpy.max(x)
numpy.max(x) - numpy.min(x)
numpy.var(x)   # 분산
numpy.std(x)  # 표준편차
numpy.sqrt(numpy.var(x))  # numpy.std(x)와 같음
numpy.sqrt(4)

numpy.quantile(x, .25)   # 1사분위수
numpy.quantile(x, .5)    # 2사분위수 = 중앙값
numpy.median(x)   # 중앙
numpy.quantile(x, .75)   # 3사분위수
numpy.quantile(x, .75) - numpy.quantile(x, .25)   # 사분위간 범

x = [8, 3, 6, 6, 9, 4, 3, 9, 3, 4]
y = [6, 2, 4, 6, 10, 5, 1, 8, 4, 5]
import matplotlib.pyplot as plt
plt.plot(x, y, 'o')
import numpy as np
np.cov(x, y)
np.cov(x, y)[0, 1]

z = [-3, -2, -1, 0, 1, 2, 3]
w = [9, 4, 1, 0, 1, 4, 9]
np.cov(z, w)[0, 1]

# 피어슨 상관계수
import  numpy     as np
np.corrcoef(x, y)
np.corrcoef(x, y)[0, 1]

cov = np.cov(x, y)[0, 1] # 공분산
xsd  =  np.std(x, ddof=1)     # x의 표본표준편차
ysd  =  np.std(y, ddof=1)     # y의 표본표준편차
cov /( xsd * ysd )

np.corrcoef(z, w)[0, 1]

import scipy.stats
scipy.stats.spearmanr(x, y).correlation
scipy.stats.kendalltau(x, y).correlation

x = [8, 3, 6, 6, 9, 4, 3, 9, 3, 4]
y = [6, 2, 4, 6, 10, 5, 1, 8, 4, 5]
import scipy.stats
scipy.stats.pearsonr(x, y)

import pandas as pd
df = pd.read_csv('cars.csv')
df.head()
import seaborn as sns
sns.regplot('speed', 'dist', lowess=True, data = df)
import matplotlib.pyplot as plt
# 1행 2열 형태로 2개의 그래프를 그린다
fig, (ax1, ax2)  =  plt.subplots(1, 2)

# speed의 상자 그림을 첫번째(ax1)로 그린다. 방향은 수직(orient='v') 
sns.boxplot('speed', data=df, ax=ax1, orient='v')  
ax1.set_title('Speed')

# dist의 상자 그림을 두번째(ax2)로 그린다.  
sns.boxplot('dist', data=df, ax=ax2, orient='v')  
ax2.set_title('Distance')

# 1행 2열 형태로 2개의 그래프를 그린다
fig, (ax1, ax2)  =  plt.subplots(1, 2)

# speed의 밀도 플롯  
sns.kdeplot(df['speed'], ax=ax1)  
ax1.set_title('Speed')

# dist의 밀도 플롯  
sns.kdeplot(df['dist'], ax=ax2)  
ax2.set_title('Distance')

import scipy.stats
scipy.stats.skew(df['speed'])
scipy.stats.skew(df['dist'])

df = pd.read_csv('cars.csv')
from statsmodels.formula.api import ols
res = ols('dist ~ speed', data=df).fit()
res.summary()

import pandas as pd
cars = pd.read_csv('cars.csv')
from statsmodels.formula.api import ols  
res = ols('dist ~ 0 + speed', cars).fit()  
res.summary()

import pandas as pd
df = pd.read_csv('crab.csv')  
df.head()

from statsmodels.formula.api import ols
model = ols('y ~ sat + weight + width', df)
res = model.fit()
res.summary()

from statsmodels.stats.outliers_influence import variance_inflation_factor
model.exog_names
variance_inflation_factor(model.exog, 1)
variance_inflation_factor(model.exog, 2)

pd.DataFrame({'컬럼': column, 'VIF': variance_inflation_factor(model.exog, i)} 
             for i, column in enumerate(model.exog_names)
             if column != 'Intercept') # 절편의 VIF는 구하지 않는다.

model = ols('y ~ sat + weight', df)
model.fit().summary()


import pandas as pd
from statsmodels.formula.api import ols


df = pd.read_csv('cars.csv')
res = ols('dist ~ speed', data=df).fit()

import matplotlib.pyplot as plt  
import seaborn as sns

fitted = res.predict(df)  
residual = df['dist'] - fitted

sns.regplot(fitted, residual, lowess=True, line_kws={'color': 'red'})  
plt.plot([fitted.min(), fitted.max()], [0, 0], '--', color='grey')

import scipy.stats
sr = scipy.stats.zscore(residual)  
(x, y), _ = scipy.stats.probplot(sr)
sns.scatterplot(x, y)
plt.plot([-3, 3], [-3, 3], '--', color='grey')

scipy.stats.shapiro(residual)

import numpy as np
sns.regplot(fitted, np.sqrt(np.abs(sr)), lowess=True, line_kws={'color': 'red'})

from statsmodels.stats.outliers_influence import OLSInfluence
cd, _ = OLSInfluence(res).cooks_distance
cd.sort_values(ascending=False).head()

dat_M = [117, 108, 105, 89, 101, 93, 96, 108, 108, 94, 93, 112, 92, 91, 100, 96, 120, 86, 96, 95]
dat_F = [121, 101, 102, 114, 103, 105, 101, 131, 96, 109, 109, 113, 115, 94, 108, 96, 110, 112, 120, 100]
import numpy as np
np.mean(dat_M)
np.mean(dat_F)
import scipy.stats
scipy.stats.ttest_ind(dat_M, dat_F, equal_var=False)

dat_M = [117, 108, 105, 89, 101, 93, 96, 108, 108, 94, 93, 112, 92, 91, 100, 96, 120, 86, 96, 95]
dat_F = [121, 101, 102, 114, 103, 105, 101, 131, 96, 109, 109, 113, 115, 94, 108, 96, 110, 112, 120, 100]
import scipy.stats
m = scipy.stats.ttest_ind(dat_M, dat_F, equal_var=False)

import numpy as np
t = m.statistic
df = len(dat_M) + len(dat_F) - 2
abs(t) / np.sqrt(df)
t2 = t ** 2
np.sqrt(t2 / (t2 + df))

dat_M = [117, 108, 105, 89, 101, 93, 96, 108, 108, 94, 93, 112, 92, 91, 100, 96, 120, 86, 96, 95]
dat_F = [121, 101, 102, 114, 103, 105, 101, 131, 96, 109, 109, 113, 115, 94, 108, 96, 110, 112, 120, 100]
import scipy.stats
scipy.stats.ttest_rel(dat_M, dat_F)











