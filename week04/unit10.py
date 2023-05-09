#!/usr/bin/env python
# coding: utf-8

# In[5]:


import csv
f = open('gender.csv', encoding='cp949')  ### gender.csv
data = csv.reader(f)
m = []
f = []
name = input('궁금한 동네를 입력해주세요 : ')
for row in data :
    if name in row[0] :
        print(name+'동 포함된 지역 :', row[0])
        for i in range(3,104) : 
            m.append(int(row[i]))
            f.append(int(row[i+103]))        
        break
import matplotlib.pyplot as plt

plt.plot(m, label = 'Male')
plt.plot(f, label = 'Female')
plt.legend()
plt.show()


# In[7]:

# 남성 인구 - 여성 인구 막대 그래프로 표현
import csv
f = open('gender.csv', encoding='cp949')  ### gender.csv
data = csv.reader(f)
result = []
name = input('궁금한 동네를 입력해주세요 : ')
for row in data :
    if name in row[0] :
        print(name+'동 포함된 지역 :', row[0])
        for i in range(3,104) : 
            result.append(int(row[i]) - int(row[i+103]))  ###
        break
import matplotlib.pyplot as plt
plt.bar(range(101), result)
plt.show()


# In[4]:

# 산점도로 표현
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.scatter([1,2,3,4], [10,30,20,40])
plt.show()


# In[5]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.scatter([1,2,3,4], [10,30,20,40], s = [100,200,250,300])
plt.show()


# In[6]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.scatter([1,2,3,4], [10,30,20,40], s = [30,60,90,120], c = ['red','blue','green','gold'])
plt.show()


# In[10]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.scatter([1,2,3,4], [10,30,20,40], s = [30,60,90,120], c = range(4))
plt.colorbar()  # 컬러바 추가
plt.show()


# In[11]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.scatter([1,2,3,4], [10,30,20,40], s = [30,60,90,120], c =range(4), cmap='jet')
plt.colorbar()
plt.show()


# In[12]:

# 위치, 크기가 랜덤인 100개의 점
import matplotlib.pyplot as plt
import random
x = []
y = []
size = []
for i in range(100) : 
    x.append(random.randint(50,100))
    y.append(random.randint(50,100))
    size.append(random.randint(10,100))
    
plt.style.use('ggplot')
plt.scatter(x, y, s = size)  
plt.show()


# In[13]:


import matplotlib.pyplot as plt
import random
x = []
y = []
size = []
for i in range(100): 
    x.append(random.randint(50,100))
    y.append(random.randint(50,100))
    size.append(random.randint(10,100))
plt.style.use('ggplot')
plt.scatter(x, y, s = size)
plt.show()


# In[14]:


plt.scatter(x, y, s = size, c = size, cmap='jet')
plt.colorbar()
plt.show()


# In[15]:


plt.scatter(x, y, s = size, c = size, cmap='jet', alpha =0.7)
plt.colorbar()
plt.show()


# In[7]:


import csv
f = open('week04/gender.csv', encoding='cp949')  ### gender.csv
data = csv.reader(f)
m = []
f = []
name = input('궁금한 동네를 입력해주세요 : ')
for row in data :
    if name in row[0] :
        print(name+'동 포함된 지역 :', row[0])
        for i in range(3,104) : 
            m.append(int(row[i]))
            f.append(int(row[i+103]))        
        break
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.scatter(m, f)
plt.show()
        


# In[2]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.scatter(m, f, c =range(101), alpha=0.5, cmap='jet')
plt.colorbar()
plt.plot(range(max(m)),range(max(m)), 'g')  # 추세선 추가
plt.show()


# In[8]:
    
import csv
import math
f = open('gender.csv', encoding='cp949')  ### gender.csv
data = csv.reader(f)
m = []
f = []
size = []
#name = input('궁금한 동네를 입력해주세요 : ')
name = '신도림'
for row in data :
    if name in row[0] :
        print(name+'동 포함된 지역 :', row[0])
        for i in range(3,104) : 
            m.append(int(row[i]))
            f.append(int(row[i+103]))
            size.append(math.sqrt(int(row[i]) + int(row[i+103])))
        break
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rc('font',family='Malgun Gothic')
plt.figure(figsize = (10,5), dpi=300)            
plt.title(name+' 지역의 성별 인구 그래프')
plt.scatter(m, f, s = size, c = range(101), alpha=0.5, cmap='jet')
plt.colorbar()
plt.plot(range(max(m)),range(max(m)), 'g')
plt.xlabel('남성 인구 수')
plt.ylabel('여성 인구 수')
plt.show()


# In[ ]:




