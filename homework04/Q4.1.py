# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 18:27:09 2022

@author: ing06
"""

import csv
import matplotlib.pyplot as plt

# 한글 폰트 깨짐 방지
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.ttc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)


# 2014년 데이터 읽기
f = open('homework04/201403.csv', encoding='cp949')
data = csv.reader(f)
next(data)

result = []
areaName = []

for row in data:
    result.append(int(row[1].replace(',', '')))
    areaName.append(row[0].split()[0])

# 2021년 데이터 읽기
f2 = open('homework04/202103.csv', encoding='cp949')
data2 = csv.reader(f2)
next(data2)

result2 = []
for row in data2:
    result2.append(int(row[1].replace(',', '')))

# 2021-2014 결과
new_result = []
for i in range(18):
    new_result.append(result2[i]-result[i])


# 막대그래프
bar = plt.bar(range(18), new_result)
# 그래프에 값 표시
for rect in bar: # 막대그래프의 막대를 가져옴
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.f' % height, ha='center', va='bottom', size = 10, color='red')
plt.xticks(range(18), labels=areaName, rotation=90)
plt.show()