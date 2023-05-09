from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from IPython.display import Image

import pandas as pd
import numpy as np
import pydotplus
import os
#pydotplus 설치 
#anaconda prompt 를 관리자 권한으로 실행
#conda install -c conda-forge pydotplus
tennis_data = pd.read_csv('playtennis.csv')
tennis_data
# 범주형 변수 변경- replace로 하나하나....아이고
tennis_data.Outlook = tennis_data.Outlook.replace('Sunny', 0)
tennis_data.Outlook = tennis_data.Outlook.replace('Overcast', 1)
tennis_data.Outlook = tennis_data.Outlook.replace('Rain', 2)

tennis_data.Temperature = tennis_data.Temperature.replace('Hot', 3)
tennis_data.Temperature = tennis_data.Temperature.replace('Mild', 4)
tennis_data.Temperature = tennis_data.Temperature.replace('Cool', 5)

tennis_data.Humidity = tennis_data.Humidity.replace('High', 6)
tennis_data.Humidity = tennis_data.Humidity.replace('Normal', 7)

tennis_data = tennis_data.drop(columns=['Wind'])

#tennis_data.Wind = tennis_data.Wind.replace('Weak', 8)
#tennis_data.Wind = tennis_data.Wind.replace('Strong', 9)

tennis_data.PlayTennis = tennis_data.PlayTennis.replace('No', 8)
tennis_data.PlayTennis = tennis_data.PlayTennis.replace('Yes', 9)

tennis_data

X = np.array(pd.DataFrame(tennis_data, columns = ['Outlook', 'Temperature', 'Humidity']))
y = np.array(pd.DataFrame(tennis_data, columns = ['PlayTennis']))
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y)
# train, test 비율은 default=0.25 로 랜덤하게 선택됨 => 아래 링크 참조.
# http://blog.naver.com/PostView.nhn?blogId=siniphia&logNo=221396370872
print(X_train)
print(X_test)
dt_clf = DecisionTreeClassifier()
dt_clf = dt_clf.fit(X_train, y_train)
dt_prediction = dt_clf.predict(X_test)
dt_prediction
print(confusion_matrix(y_test, dt_prediction))
print(classification_report(y_test, dt_prediction))

#os.environ['PATH'] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
feature_names = tennis_data.columns.tolist()
feature_names = feature_names[0:3]
target_name = np.array(['Play No', 'Play Yes'])
dt_dot_data = tree.export_graphviz(dt_clf, out_file = None, 
                                   feature_names = feature_names,
                                   class_names = target_name,
                                   filled = True, rounded = True,
                                   special_characters = True)
dt_graph = pydotplus.graph_from_dot_data(dt_dot_data)
Image(dt_graph.create_png())
