#!/usr/bin/env python
# coding: utf-8

# In[6]:


# seoul.csv 파일은 기상청(https://data.kma.go.kr)에서 받을 수 있음
import csv 
f = open('week04/seoul.csv', 'r', encoding='cp949') 
data = csv.reader(f, delimiter=',') 
for row in data :
    print(row)
f.close() 


# In[3]:
# header 출력

import csv
f =open('week04/seoul.csv', 'r', encoding='cp949') 
data = csv.reader(f)
header = next(data)  #①
print(header)        #②
f.close()


# In[4]:

# header 제외하고 출력

import csv
f =open('week04/seoul.csv')
data = csv.reader(f)
header =next(data) # next()는 데이터 위치를 다음으로 이동...
for row in data :
    print(row)
f.close()


# In[2]:


# pandas 를 사용하여 데이터 다루기..... 비교해보자
# -*- coding: euc-kr -*-
import pandas as pd
datapd = pd.read_csv("week04/seoul.csv",  encoding='cp949')
datapd.head()
print(datapd)  # 반복문없이 파일 전체 출력
print(datapd[0:5])


# In[4]:


# pandas 를 사용하여 데이터 다루기..... 비교해보자
# -*- coding: euc-kr -*-
import pandas as pd
datapd = pd.read_csv("week04/seoul.csv",  encoding='ANSI')
datapd.head()
print(datapd)
print(datapd[0:5])

datapd.describe()
# In[ ]:




