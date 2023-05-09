data_dic = {'year':[2018, 2019, 2020], 'sales':[350, 480, 1099]}

import pandas as pd
df1 = pd.DataFrame(data_dic)

print(df1)

df2 = pd.DataFrame([[2018, 350], [2019, 480], [2020, 1099]], columns=['year', 'sales'])
print(df2)

df1.index
df1.columns

df1.head(2)  # 앞 2행만 보여줌
df1.tail(2)  # 뒤 2행만 보여줌

df = pd.DataFrame([[60, 61, 61], [70, 71, 72], [80, 81, 82], [90, 91, 92]], index = ['1반', '2반', '3반', '4반'], columns = ['퀴즈1', '퀴즈2', '퀴즈3'])

# 퀴즈2 컬럼 추출
df['퀴즈2']

# 퀴즈2 컬럼의 2행 추
df['퀴즈2'][2]  
type(df)
type(df['퀴즈2'])

# loc
df.loc['2반']['퀴즈2']
df.loc['2반':'4반', '퀴즈2']
df.loc['2반':'4반', '퀴즈2':'퀴즈3']

# iloc
df.iloc[2]  # 2행 추출
df.iloc[2, 1]
df.iloc[2:3, 1]
df.iloc[2:4, 1:3]

# 3p
data = pd.DataFrame([[89.2, 92.5, 'B'], [90.8, 92.8, 'A'],
                     [89.9, 95.2, 'A'], [89.9, 85.2, 'C'], [89.9, 90.2, 'B']],
                    columns=['중간고사', '기말고사', '성적'],
                    index=['1반', '2반', '3반', '4반', '5반'])

data.describe()
data.중간고사.describe()
data.중간고사.unique()

data.groupby('중간고사').중간고사.count()
data.groupby('중간고사').중간고사.min()

import numpy as np
data.loc['6반'] = [10, 10, np.nan]
data[pd.isnull(data.성적)]

data = data.rename(columns={'성적':'등급'})
data.rename_axis('반이름', axis='rows')

data1 = pd.DataFrame([[89.2, 92.5, 'B'], [90.8, 92.8, 'A'],
                     [89.9, 95.2, 'A'], [89.9, 85.2, 'C'], [89.9, 90.2, 'B']],
                    columns=['중간고사', '기말고사', '성적'],
                    index=['1반', '2반', '3반', '4반', '5반'])

data0 = pd.concat([data, data1])
data0.to_csv("성적.csv", encoding='utf-8-sig')

data3 = pd.read_csv("성적.csv", encoding='utf-8-sig')
