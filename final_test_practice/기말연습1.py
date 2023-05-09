
""" 2주차 연습문제 """
# Q2.1 - csv 파일을 읽어서 리스트로 만들어라
import csv

with open('../emp.csv', encoding='utf-8') as f:
    reader = csv.reader(f)
    emp = list(reader)
print(emp)

import pandas as pd

emp_df = pd.read_csv('../emp.csv')
emp_df = pd.DataFrame(emp_df, columns=['EMPNO', 'ENAME', 'JOB', 'MGR', 'HIREDATE', 'SAL', 'COMM', 'DEPTNO'])

# df.columns.values.tolist(): 컬럼명을 리스트로
# df.values.tolist(): 데이터들을 리스트로
emp2 = emp_df.columns.values.tolist() + emp_df.values.tolist()
print(emp2)


""" 3주차 연습문제 """
# Q3.1
# emp.csv를 dataframe으로 만들고 출력
import pandas as pd
emp = pd.read_csv('../emp.csv')
print(emp)

# emp에서 ENAME, SAL컬럼만 출력
emp[['ENAME', 'SAL']]

# JOB컬럼에서 중복을 제거하고 출력 - series.drop_duplicates()
emp['JOB'].drop_duplicates()
set(emp['JOB'])

# SAL < 20000
emp[emp['SAL']<20000]

# 1000 < SAL < 2000
emp[(1000 < emp['SAL']) & (emp['SAL'] < 2000)]

# sal >= 1500 이고 job= ‘SALESMAN’
emp[(1500 <= emp['SAL']) & (emp['JOB'] == 'SALESMAN')]


# job이 MANAGER, CLERK인 경우 - series.isin()
emp[emp['JOB'].isin(['MANAGER', 'CLERK'])]

# ename이 BLAKE인 ename, job컬럼
emp[['ENAME', 'JOB']][emp['ENAME'] == 'BLAKE']

# ename에 AR이 포함된 ename, job컬럼 - series.str.contains('AR')
emp[['ENAME', 'JOB']][emp['ENAME'].str.contains('AR')]


# ename으로 정렬 - df.sort_values()
emp.sort_values(by='ENAME')


## 집계함수
# SAL의 sum
emp['SAL'].sum()

# job이 SALESMAN인 SAL의 sum
emp['SAL'][emp['JOB'] == 'SALESMAN'].sum()

# SUM(sal), AVG(sal), MIN(sal), MAX(sal)
emp['SAL'].agg(['sum', 'mean', 'min', 'max'])

# SELECT COUNT(*) FROM EMP - df.count()
emp.count()

# SELECT COUNT(*), SUM(sal) FROM Emp GROUP BY job;음

# SELECT * FROM Emp WHERE comm IS NOT NULL;
emp[~emp['COMM'].isnull()]

# 이외에도 df.drop(), df.rename() 등이 있

