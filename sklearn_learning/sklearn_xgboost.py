# Xgboost模型
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

titanic = pd.read_csv(
    'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

y = titanic['survived']
X = titanic.drop(['row.names', 'name', 'survived'], axis=1)
X.columns

X['age'].fillna(X['age'].mean(), inplace=True)
X.fillna('UNKNOWN', inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=33)

vec = DictVectorizer()
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

vec.feature_names_
print(len(vec.feature_names_))

# 使用随机森林进行预测
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
print('RandomForestClassifier:', rfc.score(X_test, y_test))
# RandomForestClassifier: 0.8571428571428571

# 使用Xgboost进行预测
xgbc = XGBClassifier()
xgbc.fit(X_train, y_train)
print('XGBClassifier:', xgbc.score(X_test, y_test))
# XGBClassifier: 0.8389057750759878
