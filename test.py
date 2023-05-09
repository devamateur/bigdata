import pandas as pd

data1 = [10, 20, 30, 40]
data2 = ['1반', '2반', '3반', '4반', '5반']

sr1 = pd.Series(data1)
type(sr1)
print(sr1)

sr2 = pd.Series(data1, index=[101, 102, 103, 104])
print(sr2)
sr2[102]

# matplotlib
import matplotlib.pyplot as plt

x = [2016, 2017, 2018, 2019, 2020]
y = [350, 410, 520, 695, 543]

plt.plot(x, y)
plt.title('Annual sales')
plt.xlabel('years')
plt.ylabel('sales')
plt.show()

# Q1
f = open('emp.csv', 'r')
while True:
    line = f.read()
    if not line: break
    print(line)
    
# Q2
import pandas as pd
dir(pd)
print(pd.read_csv.__doc__)
help(pd.read_csv)

dir(pd.DataFrame)
print(pd.DataFrame.dropna.__doc__)
help(pd.DataFrame.dropna)

# Q3
import pandas as pd
emp=pd.read_csv('emp.csv') 

emp.head()
emp["ENAME"]   # type(emp["ENAME]): Series
emp[emp["SAL"] >2000]
