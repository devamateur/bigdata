# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 22:32:51 2021

@author: Park
"""
import seaborn as sns
import pandas as pd
titanic = sns.load_dataset("titanic")
titanic.to_csv('./DATA/titanic.csv', index = False)

titanic.isnull().sum()   # 각 컬럼별 null값의 합을 보여줌

# null값 처리
titanic['age'] = titanic['age'].fillna(titanic['age'].median())   # age컬럼의 null값을 중앙값으로 대체함
titanic['embarked'].value_counts()
titanic['embark_town'] = titanic['embark_town'].fillna('Southampton')
titanic['deck'].value_counts()
titanic['deck'] = titanic['deck'].fillna('C')
titanic.isnull().sum()

titanic.info()
titanic.survived.value_counts()

import matplotlib.pyplot as plt

# 성별별 생존 여부 - 여성 생존율이 높음
f,ax = plt.subplots(1, 2, figsize = (10, 5))
titanic['survived'][titanic['sex'] == 'male'].value_counts().plot. \
    pie(explode = [0,0.1], autopct = '%1.1f%%', ax = ax[0], shadow = True)
titanic['survived'][titanic['sex'] == 'female'].value_counts().plot. \
       pie(explode = [0,0.1], autopct = '%1.1f%%', ax = ax[1], shadow = True)
ax[0].set_title('Survived (Male)')
ax[1].set_title('Survived (Female)')
plt.show()

# 등급별 생존 여부 - 1등급 탑승객의 생존율이 더 높음
sns.countplot('pclass', hue = 'survived', data = titanic)
plt.title('Pclass vs Survived')
plt.show()

# 상관관계 출력
titanic_corr = titanic.corr(method = 'pearson')
titanic_corr
titanic_corr.to_csv('DATA/titanic_corr.csv', index = False)

titanic['survived'].corr(titanic['adult_male'])
titanic['survived'].corr(titanic['fare'])

titanic['adult_male'] = titanic['adult_male'].replace({'True':1, 'False':0}).astype('int64')
titanic['alone'] = titanic['alone'].replace({'True':1, 'False':0}).astype('int64')

#pairplot_data =  titanic[['pclass', 'age', 'sibsp', 'parch', 'fare', 'adult_male', 'alone', 'survived']]
sns.pairplot(titanic, hue='survived')
#g = sns.PairGrid(titanic, hue = 'survived')
#g.map_diag(sns.kdeplot)
#g.map_offdiag(sns.scatterplot)
plt.show()


# 성별별 등급, 생존여부
sns.catplot(x = 'pclass', y = 'survived', hue = 'sex', data = titanic, kind = 'point')
plt.show()

def category_age(x):
        if x < 10:
           return 0
        elif x < 20:
           return 1
        elif x < 30:
           return 2
        elif x < 40:
           return 3
        elif x < 50:
            return 4
        elif x < 60:
           return 5
        elif x < 70:
           return 6
        else:
           return 7

titanic['age2'] = titanic['age'].apply(category_age)
titanic['sex'] = titanic['sex'].map({'male':1, 'female':0})
titanic['family'] = titanic['sibsp'] + titanic['parch'] + 1   # 가족 구성원 여부
titanic.to_csv('./DATA/titanic3.csv', index = False)
heatmap_data = titanic[['survived', 'sex', 'age2', 'family', 'pclass', 'fare']]
colormap = plt.cm.RdBu
# 각 변수들끼리의 상관관계를 heatmap으로 표현
sns.heatmap(heatmap_data.astype(float).corr(), linewidths = 0.1, vmax
        = 1.0, square = True, cmap = colormap, linecolor = 'white', annot = True,
        annot_kws = {"size": 10})
plt.show()
