""" 4주차 """

# 수업
# 1. seoul.csv에서 최고기온 찾기
import csv
f = open('../week04/seoul.csv', encoding='cp949')
data = csv.reader(f)
header = next(data)   # 헤더만 저장

max_value = -999
max_date = ''
for row in data:
    if row[-1] == '':   # 값이 비어있는 경우 임의로 값 넣어줌
        row[-1] = -999
    row[-1] = float(row[-1])
    if row[-1] > max_value:  # 해당 데이터의 최고 기온 값이 더 크면
        max_value = row[-1]
        max_date = row[0]

f.close()

print("서울 최고 기온이 가장 높은 날은 ", max_date + '로, ', max_value,'도 였습니다.')


# 2. 1월과 8월의 최고 기온을 히스토그램으로 표현
import csv
f = open('../week04/seoul.csv', encoding='cp949')
data = csv.reader(f)
header = next(data)

jan = []
aug = []

for row in data:
    month = row[0].split('-')[1]  # 날짜 컬럼에서 month만 가져옴
    if row[-1] != '':
        if month == '01':
            jan.append(float(row[-1]))
        if month == '08':
            aug.append(float(row[-1]))
f.close()

import matplotlib.pyplot as plt
plt.hist(jan, bins = 100, color='b', label='Jan')  # label: 범례 내용
plt.hist(aug, bins=100, color='r', label='Aug')
plt.boxplot([jan, aug])
plt.legend()  # 범례표시
plt.show()

########### 연습문제 ###########
# 인구변동 시각화
import csv
import matplotlib.pyplot as plt

f = open('../homework04/201403.csv', encoding='cp949')
data = csv.reader(f)
next(data)

areaName = []
total_number = []

for row in data:
    areaName.append(row[0].split()[0])  # 지역 이름을 가져옴
    total_number.append(int(row[1].replace(',', '')))   # 총 인구수에서 콤마 제거
f.close()

f2 = open('../homework04/202103.csv', encoding='cp949')
data2 = csv.reader(f2)
next(data2)

total_number2 = []
for row in data2:
    total_number2.append(int(row[1].replace(',', '')))
f2.close()

# 두 년도의 인구수 차이 구하기
result = []

for i in range(len(total_number)):
    result.append(total_number2[i] - total_number[i])

plt.style.use('default')
plt.rc('font',family='Malgun Gothic')  # 한글폰트 깨짐 방지
bar = plt.bar(range(18), result)  # 막대 그래프

# 막대 그래프 위에 인구수 표시
for rect in bar:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.f'%height, ha='center', va='bottom'
             , size=10, color='red')
plt.xticks(range(18), labels = areaName, rotation = 90)
plt.show()

### 2. 인구변동 시각화 - pandas ###
import pandas as pd

df1 = pd.read_csv('../homework04/201403.csv', encoding='cp949')
df1['행정구역'] = df1['행정구역'].str.split().str[0]
df1['2014년03월_계_총인구수'] = df1['2014년03월_계_총인구수'].str.replace(',', '')

df2 = pd.read_csv('../homework04/202103.csv', encoding='cp949')
df2['행정구역'] = df2['행정구역'].str.split().str[0]
df2['2021년03월_계_총인구수'] = df2['2021년03월_계_총인구수'].str.replace(',','')

df1 = df1.astype({'2014년03월_계_총인구수':int})
df2 = df2.astype({'2021년03월_계_총인구수':int})

result = df2['2021년03월_계_총인구수'] - df1['2014년03월_계_총인구수']

import matplotlib.pyplot as plt
bar = plt.bar(range(18), result)
for rect in bar:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.f'%height, ha='center', va='bottom'
             , size=10, color='red')
plt.xticks(range(18), labels = df1['행정구역'], rotation = 90)
plt.show()


""" 5주차 """
# 수업
# 7시~9시 승차인원이 가장 많은 역 찾기 - pandas
import pandas as pd

df = pd.read_csv('../week05/subwaytime.csv', encoding='cp949')

# df.iloc[행, 열]
mydata=pd.concat([df.iloc[1:, 0:4], df.iloc[1:, 4:53:2].astype(int)], axis=1)
mydata.info()
mydata.fillna(0) 

mydata['sum'] = mydata[['07:00:00~07:59:59', '08:00:00~08:59:59', '09:00:00~09:59:59']].sum(axis=1)
k=mydata['sum'].idxmax()   # 최대값을 가지는 데이터의 인덱스
mydata['지하철역'][k]


########### 연습문제 ###########
# 5.1 - 서울시 지하철 승하차 인원 비교
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 전처리
def data_processing(csvPath):
    df = pd.read_csv(csvPath, header=None)
    df = df[2:]
    
    geton = df.iloc[:, 4:52:2]
    geton = geton.apply(lambda x: x.str.replace(',','').astype('int64'), axis=1)
    
    getoff = df.iloc[:, 5:52:2]
    getoff = getoff.apply(lambda x: x.str.replace(',','').astype('int64'), axis=1)
    
    return geton, getoff
    

geton, getoff = data_processing('../homework05/201903_traffic.csv')
geton2, getoff2 = data_processing('../homework05/202003_traffic.csv')
geton3, getoff3 = data_processing('../homework05/202103_traffic.csv')
geton4, getoff4 = data_processing('../homework05/202203_traffic.csv')


plt.style.use('default')
plt.rc('font', family='Malgun Gothic')
plt.plot(range(24), geton.sum(), label='201903승차')
plt.plot(range(24), getoff.sum(), label='201903하차')
plt.plot(range(24), geton2.sum(), label='202003승차')
plt.plot(range(24), getoff2.sum(), label='202003하차')
plt.plot(range(24), geton3.sum(), label='202103승차')
plt.plot(range(24), getoff3.sum(), label='202103하차')
plt.plot(range(24), geton4.sum(), label='202203승차')
plt.plot(range(24), getoff4.sum(), label='202203하차')
plt.legend()
plt.xticks(range(24), range(4, 28))
plt.show()


# Q5.2 - 강남구 대치1동과 인구구조가 가장 비슷한 5곳 찾기
# 하기 싫어요~
