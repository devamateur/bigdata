import pandas as pd
import matplotlib.pyplot as plt


# csv파일을 읽고 
# 승/하차 데이터에서 콤마 제거 및 int형변환
def data_processing(filePath):
    df = pd.read_csv(filePath, header=None)
    df = df[2:]
    
    geton = df.iloc[:, 4:52:2]
    geton = geton.apply(lambda x: x.str.replace(',', '').astype('int64'), axis=1)
    
    getoff = df.iloc[:, 5:52:2]
    getoff = getoff.apply(lambda x: x.str.replace(',','').astype('int64'), axis=1)
    
    return geton, getoff

geton, getoff = data_processing('201903_traffic.csv')
geton2, getoff2 = data_processing('202003_traffic.csv')
geton3, getoff3 = data_processing('202103_traffic.csv')
geton4, getoff4 = data_processing('202203_traffic.csv')

plt.style.use('default')
#plt.figure(dpi = 300)
plt.rc('font', family = 'Malgun Gothic')
plt.title('지하철 시간대별 승하차 인원 추이(단위 1000만명)')
plt.plot(range(24), geton.sum(), label = '201903승차')
plt.plot(range(24), getoff.sum(), label = '201903하차', linestyle=':')
plt.plot(range(24), geton2.sum(), label = '202003승차')
plt.plot(range(24), getoff2.sum(), label = '202003하차', linestyle=':')
plt.plot(range(24), geton3.sum(), label = '202103승차')
plt.plot(range(24), getoff3.sum(), label = '202103하차', linestyle=':')
plt.plot(range(24), geton4.sum(), label = '202203승차')
plt.plot(range(24), getoff4.sum(), label = '202203하차', linestyle=':')
plt.legend()
plt.xticks(range(24), range(4, 28))
plt.show()