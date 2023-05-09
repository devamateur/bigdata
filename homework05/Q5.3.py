import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('age.csv', encoding='cp949', header=None)
df = df[1:]

# 동 이름 입력받기
name = input('원하는 지역의 이름(읍면동 단위)을 입력해주세요 : ')

new_areaname = df.iloc[:, 0].str.split(' ').tolist()  # 행정구역
dong = []  # 동 이름만 저장
home = [] # 입력받은 지역의 데이터 저장
away = [] # 다른 지역들의 데이터 저장
result_list = [] # 최종적으로 가장 비슷한 지역들을 저장

for i in range(len(new_areaname)):
    dong.append(new_areaname[i][2].split('('))

    # 저장된 동 이름과 입력받은 이름이 같으면 home에 저장
    if(dong[i][0] == name):
        home.append(df.iloc[i, 3:].astype('int64').values.tolist())    #입력받은 지역의 0세~100세이상    
        hometotal = int(df.iloc[i, 2])   # 연령구간 인구

# 입력받은 동의 인구비율 계산
for k in range(len(home)):
    for l in range(len(home[k])):
        home[k][l]=(home[k][l]/hometotal)

home = np.concatenate(home).tolist()  # 1차원 배열로 

# 다른 지역의 인구비율 계산
away = df.iloc[:, 3:].astype('int64').values.tolist()
awaytotal = df.iloc[:, 2].astype('int64').tolist()

for k in range(len(away)):
    for l in range(len(away[k])):
        away[k][l] = (away[k][l]/awaytotal[k])

# 인구비율이 비슷한 지역 찾기
for j in range(len(away)):
    sum=0
    for k in range(len(home)):
        sum += (home[k]-away[j][k])**2   # 입력받은 지역과 다른 지역의 인구비율차이의 제곱을 더함
    result_list.append([df.iloc[j, 0], away[j], sum])  # 지역 이름, 인구비율, sum
result_list.sort(key=lambda sum: sum[2]) # sum 값으로 정렬(람다식 이용)

plt.figure(figsize = (10,5), dpi=300)            
plt.rc('font', family ='Malgun Gothic')
plt.title(name+' 지역과 가장 비슷한 인구구조를 가진 지역')
plt.plot(home, label=result_list[0][0])
plt.plot(result_list[1][1], label=result_list[1][0])
plt.plot(result_list[2][1], label = result_list[2][0])
plt.plot(result_list[3][1], label = result_list[3][0])
plt.plot(result_list[3][1], label = result_list[4][0])
plt.legend(title='행정구역')
plt.xticks([0, 20, 40, 60, 80, 100], ['0세', '20세', '40세', '60세', '80세', '100세 이상'])
plt.show()