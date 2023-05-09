import pandas as pd
import matplotlib.pyplot as plt

# 한글 폰트 깨짐 방지
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.ttc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

df1 = pd.read_csv("201403.csv", encoding='cp949')
df1['2014년03월_계_총인구수'] = df1['2014년03월_계_총인구수'].str.replace(',', '')
df1['행정구역'] = df1['행정구역'].str.split().str[0]

df2 = pd.read_csv("202103.csv", encoding='cp949')
df2['2021년03월_계_총인구수'] = df2['2021년03월_계_총인구수'].str.replace(',', '')
df2['행정구역'] = df2['행정구역'].str.split().str[0]

# astype({칼럼명:타입}) 칼럼 값의 타입을 int로 변환
df1 = df1.astype({'2014년03월_계_총인구수':int})
df2 = df2.astype({'2021년03월_계_총인구수':int})
result_df = df2['2021년03월_계_총인구수'] - df1['2014년03월_계_총인구수']

bar = plt.bar(range(18), result_df)
# 그래프에 값 표시
for rect in bar: # 막대그래프의 막대를 가져옴
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.f' % height, ha='center', va='bottom', size = 10, color='red')
plt.xticks(range(18), labels=df2['행정구역'], rotation=90)
plt.show()
