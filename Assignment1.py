import pandas as pd

# 3-1
emp = pd.read_csv("emp.csv")

# 3-2
print(emp)

# 3-3
emp['ENAME']

# 3-4
emp[['ENAME', 'SAL']]

# 3-5
emp['JOB'].drop_duplicates()

# 3-6
emp[emp['SAL'] < 20000]

# 3-7
emp[(1000<emp['SAL']) & (emp['SAL']<2000)]

# 3-8
emp[(emp['SAL']>=1500) & (emp['JOB']=='SALESMAN')]

# 3-9
emp[emp['JOB'].isin(['MANAGER', 'CLERK'])]

# 3-10
emp[~emp['JOB'].isin(['MANAGER', 'CLERK'])]

# 3-11
emp[['ENAME', 'JOB']][emp['ENAME'] == 'BLAKE']

# 3-12
emp[['ENAME', 'JOB']][emp['ENAME'].str.contains('AR')]

dir(emp['ENAME'].str)

# 3-13
emp[(emp['ENAME'].str.contains('AR')) & (emp['SAL'] >= 2000)]

# 3-14 sort 이용
emp.sort_values(by='ENAME')

# 3-15
emp['SAL'].sum()

# 3-16
emp['SAL'][emp['JOB'] == 'SALESMAN'].sum()

# 3-17
# agg()를 이용해 집계연산을 수행할 수 있
emp['SAL'].agg(['sum', 'mean', 'min', 'max'])

# 3-18
emp.count()

# 3-19
emp.groupby('JOB')['EMPNO'].count()

# 3-20
emp[~emp['COMM'].isnull()]


# 4-0
emp = pd.read_csv("emp.csv")

# 4-1
emp['age'] = [30,40,50,30,40,50,30,40,50,30,40,50,30,40]
emp
# 4-2
new_data = {'EMPNO': 9999, 'ENAME': 'ALLEN', 'JOB': 'SALESMAN'}
emp = emp.append(new_data, ignore_index=True)

# 4-3
emp = emp[~(emp['ENAME']=='ALLEN')]
emp

# 4-4
emp = emp.drop(columns = 'HIREDATE')   # emp.drop(columns = 'HIREDATE', inplace=True)
emp

# 4-5
emp[emp['ENAME'] == 'SCOTT']['SAL'] = 3000

# 5-1
emp = emp.rename(columns = {'SAL':'OLDSAL'})
emp
# 5-2
emp['NEWSAL'] = emp['OLDSAL']
emp
# 5-3
emp = emp.drop(columns = 'OLDSAL')
emp