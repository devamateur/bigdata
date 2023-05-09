import pandas as pd

# csv 파일을 읽어온다 ;를 구분자로 지정
red_df = pd.read_csv('./Data/winequality-red.csv', sep = ';', header = 0, engine = 'python')
white_df = pd.read_csv('./DATA/winequality-white.csv', sep = ';', header = 0, engine= 'python')
red_df.to_csv('./DATA/winequality-red2.csv', index = False)
white_df.to_csv('./DATA/winequality-white2.csv', index = False)

red_df.head()
red_df.insert(0, column = 'type', value = 'red')   # 컬럼을 추가(0번째 컬럼에, type명 컬럼)
red_df.head()
red_df.shape
white_df.head()
white_df.insert(0, column = 'type', value = 'white')
white_df.head()
white_df.shape
wine = pd.concat([red_df, white_df])
wine.shape
wine.to_csv('./DATA/wine.csv', index = False)

wine.info()

wine.columns = wine.columns.str.replace(' ', '_')   # 컬럼명에 있는 공백을 _으로 대체
wine.head()
wine.describe()   # count, mean, std, min, 1, 2, 3사분위 수, max값이 나옴

sorted(wine.quality.unique())   # wine quality를 정렬된 형태로 반환
wine.quality.value_counts()

wine.groupby('type')['quality'].describe()   # 레드/화이트 와인의 quality에 대한 통계를 보여줌
wine.groupby('type')['quality'].mean()       
wine.groupby('type')['quality'].std()
wine.groupby('type')['quality'].agg(['mean', 'std'])


#### 선형회귀분석 ###
from scipy import stats   # t검정을 하기 위해 
from statsmodels.formula.api import ols, glm   # 회귀분석

# 레드 와인의 quality 컬럼 추출
red_wine_quality = wine.loc[wine['type'] == 'red', 'quality']

# 화이트 와인의 quality 컬럼 추출
white_wine_quality = wine.loc[wine['type'] == 'white', 'quality']

# 두 와인의 quality에 대해 t검정 진행
stats.ttest_ind(red_wine_quality, white_wine_quality, equal_var = False)

# 회귀분석
Rformula = 'quality ~ fixed_acidity + volatile_acidity + citric_acid + \
      residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + \
      density + pH + sulphates + alcohol'
regression_result = ols(Rformula, data = wine).fit()
print(regression_result.summary())

# quality와 type을 제외한 컬럼을 저장
sample1 = wine[wine.columns.difference(['quality', 'type'])]
sample1 = sample1[0:5][:]
# 예측
sample1_predict = regression_result.predict(sample1)
print(sample1_predict)
wine[0:5]['quality']

data = {"fixed_acidity" : [8.5, 8.1], "volatile_acidity":[0.8, 0.5],
"citric_acid":[0.3, 0.4], "residual_sugar":[6.1, 5.8], "chlorides":[0.055,
0.04], "free_sulfur_dioxide":[30.0, 31.0], "total_sulfur_dioxide":[98.0,
99], "density":[0.996, 0.91], "pH":[3.25, 3.01], "sulphates":[0.4, 0.35],
"alcohol":[9.0, 0.88]}
sample2 = pd.DataFrame(data, columns= sample1.columns)
sample2
sample2_predict = regression_result.predict(sample2)
sample2_predict

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')
sns.distplot(red_wine_quality, kde = True, color = "red", label = 'red wine')
sns.distplot(white_wine_quality, kde = True, label = 'white wine')
plt.title("Quality of Wine Type")
plt.legend()
plt.show()

# 죽여줘...박교수 그러는 거 아니다
# 나왔는데요? 저는 나왔는데요? 저는 잘 나오는데요?
import statsmodels.api as sm

# quality, fixed_acidity를 제외한 나머지 변수들의 집합
others = list(set(wine.columns).difference(set(["quality", "fixed_acidity"])))

# fixed_acidity와 quality의 회귀분석 결과(부분회귀플롯)
p, resids = sm.graphics.plot_partregress("quality", "fixed_acidity", others, data = wine, ret_coords = True)
plt.show()
fig = plt.figure(figsize = (8, 13))
sm.graphics.plot_partregress_grid(regression_result, fig = fig)
plt.show()