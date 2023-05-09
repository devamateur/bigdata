#!/usr/bin/env python
# coding: utf-8

# In[1]:
# 지하철 시간대별 이용현황, subwaytime.csv
import csv
f = open('subwaytime.csv', encoding='cp949')
data = csv.reader(f)
for row in data :
    print(row)


# In[2]:
# 지하철 시간대별 이용현황, subwaytime.csv
import csv
f = open('subwaytime.csv', encoding='cp949')
data = csv.reader(f)
next(data) # 한 row를 건너뜀(header를 제외하깅 위해)
next(data)
for row in data :
    row[4:] = map(int, row[4:])  # 승하차 인원을 int로
    print(row)


# In[3]:
# 지하철 시간대별 이용현황, subwaytime.csv
import csv
f = open('subwaytime.csv', encoding='cp949')
data = csv.reader(f)
next(data)
next(data)
result = []
for row in data :
    row[4:] = map(int, row[4:]) # 새벽4시부터 1시간대 04:00:00~04:59:59 ~
    result.append(row[10]) # 7시-8시 07:00:00~07:59:59 승차인원
print(len(result))
print(result)


# In[2]:
# 지하철 시간대별 이용현황, subwaytime.csv
import csv
f = open('subwaytime.csv', encoding='cp949')
data = csv.reader(f)
next(data)
next(data)
result = []
for row in data :
    row[4:] = map(int, row[4:])
    result.append(row[10]) # 7시-8시 07:00:00~07:59:59
import matplotlib.pyplot as plt # 7시-8시 지하철 
#plt.figure(dpi = 300)
plt.style.use('ggplot')
plt.bar(range(len(result)), result)
plt.show()


# In[11]:
import csv
f = open('subwaytime.csv', encoding='cp949')
data = csv.reader(f)
next(data)
next(data)
result = []
for row in data :
    row[4:] = map(int, row[4:])
    result.append(row[10]) # 7시-8시
import matplotlib.pyplot as plt
result.sort() # 오름차순 정렬, reverse=true 옵션으로 내림차순 정렬할 수 있음
#plt.figure(dpi = 300)
plt.style.use('ggplot')
plt.bar(range(len(result)), result)
plt.show()


# In[12]:
import csv
f = open('subwaytime.csv', encoding='cp949')
data = csv.reader(f)
next(data)
next(data)
result = []
for row in data :
    row[4:] = map(int, row[4:])
    result.append(sum(row[10:15:2])) # 7시-9시 승차인원  row[start:end:step]
import matplotlib.pyplot as plt
result.sort() # 정렬
#plt.figure(dpi = 300)
plt.style.use('ggplot')
plt.bar(range(len(result)), result)
plt.show()


# In[1]:
# 지하철 시간대별 이용현황, subwaytime.csv, 
# 7시-9시 가장 승차인원이 많은 역
import csv
f = open('subwaytime.csv', encoding='cp949')
data = csv.reader(f)
next(data)
next(data)
mx = 0
mx_station = ''
for row in data :
    row[4:] = map(int, row[4:])
    if sum(row[10:15:2]) > mx :
        mx = sum(row[10:15:2]) # 07:00:00~07:59:59 ~ 09:00:00~09:59:59 승차인원
        mx_station = row[3] + '(' + row[1] + ')'
print(mx_station, mx)

# 위 코드를 pandas로
import pandas as pd
mydata=pd.read_csv('subwaytime.csv', encoding='cp949')

# 승차/하차를 분리
# mydata.iloc[1:, 0:4]: 1행, 0~3열까지
# mydata.iloc[1:, 4:53:2] : 승차 인원(4, 6, 8, 10, ... 52)
data1=pd.concat([mydata.iloc[1:, 0:4], mydata.iloc[1:, 4:53:2].astype(int)], axis=1)
data1.info()
data1.fillna(0)   # null값이 있을 경우 0으로 채움

# sum이라는 컬럼을 만들고 7시~9시까지의 승차인원의 합을 저장
data1.loc[:, 'sum'] = data1.loc[:, ['07:00:00~07:59:59', '08:00:00~08:59:59', '09:00:00~09:59:59']].sum(axis=1)
data1['sum'].max()
k=data1['sum'].idxmax()
data1['지하철역'][k]
data1.loc[k, '지하철역']

# In[4]:
# 지하철 시간대별 이용현황, subwaytime.csv, 
# 7시-9시 가장 하차인원이 많은 역
import csv
f = open('subwaytime.csv', encoding='cp949')
data = csv.reader(f)
next(data)
next(data)
mx = 0
mx_station = ''
for row in data :
    row[4:] = map(int, row[4:])
    a = row[11:16:2] # 07:00:00~07:59:59 하차인원
    if sum(a) > mx :
        mx = sum(a)
        mx_station = row[3] + '(' + row[1] + ')'
print(mx_station, mx)


# In[9]:
# 승차인원이 가장 많은 역, 시간입력
import csv
f = open('subwaytime.csv', encoding='cp949')
data = csv.reader(f)
next(data)
next(data)
mx = 0
mx_station = ''
#t = int(input('몇 시의 승차인원이 가장 많은 역이 궁금하세요? : '))
t = 12
for row in data :
    row[4:] = map(int, row[4:]) 
    a = row[2 * t - 4] # 승차인원 모두  # row[4+(t-4)*2]
    if a > mx : 
        mx = a # 최대승차인원...
        mx_station = row[3] + '(' + row[1] + ')'  # 최대승차인원 역이름
print(mx_station, mx)


# In[10]:
# 시간대별로 사람들이 가장 많이 승차하는 역은?
import csv
f = open('subwaytime.csv', encoding='cp949')
data = csv.reader(f)
next(data)
next(data)
mx = [0] * 24
mx_station = [''] * 24
for row in data :
    row[4:] = map(int, row[4:]) 
    for j in range(24) :
        a = row[j * 2 + 4]
        if a > mx[j] :
            mx[j] = a  # 시간대별 최대승차인원 
            mx_station[j] = row[3]  # 시간대별 최대승차인원 역이름
print(mx_station)
print(mx)


# In[11]:

# 시간대별 사람들이 가장 많이 승차하는 역 그래프를 그림
import matplotlib.pyplot as plt
plt.rc('font', family ='Malgun Gothic')
#plt.figure(dpi = 300)
plt.bar(range(24), mx)  # 최대 승차인원
plt.xticks(range(24), mx_station, rotation =90) # 승차역이름
plt.show()


# In[21]:
# 시간대별로 사람들이 가장 많이 승차하는 역은?(x축에 시간추가)
import csv
f = open('subwaytime.csv', encoding='cp949')
data = csv.reader(f)
next(data)
next(data)
mx = [0] * 24
mx_station = [''] * 24
for row in data :
    row[4:] = map(int, row[4:])
    for j in range(24) :
        a = row[j * 2 + 4]
        if a > mx[j] :
            mx[j] = a # 시간대별 최대승차인원 
            mx_station[j] = row[3]+'('+str(j+4)+')'  # 시간대별 최대승차인원 역이름
import matplotlib.pyplot as plt
#plt.figure(dpi = 300)
plt.rc('font',family = 'Malgun Gothic')
plt.bar(range(24), mx)
plt.xticks(range(24), mx_station, rotation = 90)
plt.show()


# In[13]:
# 시간대별로 사람들이 가장 많이 하차하는 역은?
import csv
import matplotlib.pyplot as plt
f = open('subwaytime.csv', encoding='cp949')
data = csv.reader(f)
next(data)
next(data)
mx = [0] * 24
mx_station = [''] * 24
for row in data :
    row[4:] = map(int, row[4:])
    for j in range(24) :
        b = row[5 + j * 2]
        if b > mx[j] :
            mx[j] = b   # 시간대별 최대하차인원 
            mx_station[j] = row[3]+'('+str(j+4)+')'   # 시간대별 최대하차인원 역이름
#plt.figure(dpi = 300)
plt.rc('font',family = 'Malgun Gothic')
plt.bar(range(24), mx, color = 'b')
plt.xticks(range(24), mx_station, rotation = 90)
plt.show()


# In[14]:
# 지하철 시간대별 승하차 인원 추이
import csv
f = open('subwaytime.csv', encoding='cp949')
data = csv.reader(f)
next(data)
next(data)
s_in = [0] * 24
s_out = [0] * 24
for row in data :
    row[4:] = map(int, row[4:]) 
    for i in range(24) :
        s_in[i] += row[4 + i * 2] # 시간대별 승차인원
        s_out[i] += row[5 + i * 2] # 시간대별 하차인원
import matplotlib.pyplot as plt
plt.figure(dpi = 300)
plt.rc('font', family = 'Malgun Gothic')
plt.title('지하철 시간대별 승하차 인원 추이')
plt.plot(s_in, label = '승차')
plt.plot(s_out, label = '하차')
plt.legend()
plt.xticks(range(24), range(4,28))
plt.show()


# In[ ]:




