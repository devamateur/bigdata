#!/usr/bin/env python
# coding: utf-8

# In[3]:


# plot 그래프
import csv
f = open('week04/seoul.csv', encoding='cp949')
data = csv.reader(f)
next(data)
result = []

for row in data :
    if row[-1] != '' :
        result.append(float(row[-1]))

import matplotlib.pyplot as plt
#plt.figure(figsize = (10,2), dpi = 300)
plt.plot(result, 'r')
plt.show()


# In[2]:


# histogram
import matplotlib.pyplot as plt
#plt.figure(dpi = 300)
plt.hist([1,1,2,3,4,5,6,6,7,8,10]) # 각 숫자의 빈도...
plt.show()


# In[5]:

# 주사위 시뮬레이션
# random number
import random
print('*randint : ', random.randint(1,6))

for i in range(5) :
    print(random.randint(1,6))
    
dice = []
for i in range(5) :
    dice.append(random.randint(1,6))
print('*dice : ', dice)


# In[22]:


# 주사위 히스토그램
import random
dice = []
for i in range(5) :
    dice.append(random.randint(1,6))
print(dice)

import matplotlib.pyplot as plt
#plt.figure(dpi = 300)
plt.hist(dice, bins = 6)
plt.show()


# In[21]:


import random
dice = []
for i in range(100) :
    dice.append(random.randint(1,6))
#print(dice)

import matplotlib.pyplot as plt
#plt.figure(dpi = 300)
plt.hist(dice, bins = 6)
plt.show()


# In[9]:


# 10000번 주사위를 던지면...
import random
dice = []
for i in range(1000000) :
    dice.append(random.randint(1,6))

import matplotlib.pyplot as plt
#plt.figure(dpi = 300)
plt.hist(dice, bins = 6)
plt.show()


# In[24]:


# 서울의 최고기온 온도값을 histogram으로 그려본다.
import csv
f = open('week04/seoul.csv', encoding='cp949')
data = csv.reader(f)
next(data)
result = []

for row in data :
    if row[-1] != '' :
        result.append(float(row[-1]))

import matplotlib.pyplot as plt
plt.figure(dpi = 300)
plt.hist(result, bins = 100, color = 'r')
plt.show()


# In[27]:


plt.figure(dpi = 300)
plt.hist(result, bins = 1000, color = 'r')
plt.show()


# In[28]:


import csv
f = open('week04/seoul.csv', encoding='cp949')
data = csv.reader(f)
next(data)
aug = []

for row in data :
    month = row[0].split('-')[1]
    if row[-1] != '' :
        if month == '08':
            aug.append(float(row[-1]))
print(len(aug)) # 8월 전체 최고기온
import matplotlib.pyplot as plt
#plt.figure(dpi = 300)
plt.hist(aug, bins = 100, color = 'r')
plt.show()


# In[13]:


import csv
f = open('week04/seoul.csv', encoding='cp949')
data = csv.reader(f)
next(data)
aug = []
jan = []

for row in data :
    month = row[0].split('-')[1]
    if row[-1] != '' :
        if month == '08':
            aug.append(float(row[-1]))
        if month == '01':
            jan.append(float(row[-1]))

import matplotlib.pyplot as plt
#plt.figure(dpi = 300)
plt.hist(aug, bins = 100, color = 'r', label = 'Aug')
plt.hist(jan, bins = 100, color = 'b', label = 'Jan')
plt.legend()
plt.show()


# In[29]:


# boxplot
import matplotlib.pyplot as plt
#plt.figure(dpi = 300)
import random
result = []
for i in range(13) :
    result.append(random.randint(1,1000))
print(sorted(result))
#import numpy as np
#result = np.array(result)
#print("1/4: " + str(np.percentile(result,25)))
#print("2/4: " + str(np.percentile(result,50)))
#print("3/4: " + str(np.percentile(result,75)))
plt.boxplot(result)
plt.show()


# In[15]:


# boxplot - 최고기온
import csv
f = open('seoul.csv')
data = csv.reader(f)
next(data)
result = []

for row in data :
    if row[-1] != '' :
        result.append(float(row[-1]))

import matplotlib.pyplot as plt
#plt.figure(dpi = 300)
plt.boxplot(result)
plt.show()


# In[16]:


# 8월, 1월 최고기온 boxplot
import csv
f = open('seoul.csv')
data = csv.reader(f)
next(data)
aug = []
jan = []

for row in data :
    month = row[0].split('-')[1]
    if row[-1] != '' :
        if month == '08':
            aug.append(float(row[-1]))
        if month == '01':
            jan.append(float(row[-1]))

import matplotlib.pyplot as plt
#plt.figure(dpi = 300)
plt.boxplot(aug)
plt.boxplot(jan)
plt.show()


# In[17]:


# 8월, 1월 최고기온 boxplot
import csv
f = open('seoul.csv')
data = csv.reader(f)
next(data)
aug = []
jan = []

for row in data :
    month = row[0].split('-')[1]
    if row[-1] != '' :
        if month == '08':
            aug.append(float(row[-1]))
        if month == '01':
            jan.append(float(row[-1]))

import matplotlib.pyplot as plt
#plt.figure(dpi = 300)
plt.boxplot([aug,jan])
plt.show()


# In[18]:


# 12개월 최고기온 boxplot
import matplotlib.pyplot as plt
import csv
f = open('seoul.csv')
data = csv.reader(f)
next(data)
month = [[],[],[],[],[],[],[],[],[],[],[],[]]

for row in data :
    if row[-1] != '' :
        month[int(row[0].split('-')[1])-1].append(float(row[-1]))

#plt.figure(figsize=(10,5), dpi=300)
plt.boxplot(month)
plt.show()


# In[19]:


# 8월 최고기온 boxplot
import matplotlib.pyplot as plt
import csv
f = open('seoul.csv')
data = csv.reader(f)
next(data)

day = []
for i in range(31) : 
    day.append([])

for row in data :
    if row[-1] != '' :
        if row[0].split('-')[1] == '08':
            day[int(row[0].split('-')[2])-1].append(float(row[-1]))

plt.boxplot(day, showfliers=False)
plt.show()


# In[20]:


# 8월 최고기온 boxplot - 그래프 크기 변경
import matplotlib.pyplot as plt
import csv
f = open('week04/seoul.csv', encoding='cp949')
data = csv.reader(f)
next(data)

day = [[] for i in range(31)]

for row in data :
    if row[-1] != '' :
        if row[0].split('-')[1] == '08':
            day[int(row[0].split('-')[2])-1].append(float(row[-1]))
        
plt.style.use('ggplot')
plt.figure(figsize=(10,5), dpi=300)
plt.boxplot(day, showfliers=False)
plt.show()


# In[ ]:




