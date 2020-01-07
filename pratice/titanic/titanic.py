from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
import pandas as pd

train = pd.read_csv(
    '/Users/hejinyang/Desktop/machine_learning/pratice/titanic_train.csv')
test = pd.read_csv(
    '/Users/hejinyang/Desktop/machine_learning/pratice/titanic_test.csv')

print(train.info())
print(test.info())

selected_features = ['Pclass', 'Sex', 'Age',
                     'SibSp', 'Parch', 'Fare', 'Embarked']


X_train = train[selected_features]
X_test = test[selected_features]

y_train = train['Survived']

print(X_train['Embarked'].value_counts())
print(X_test['Embarked'].value_counts())

X_train['Embarked'].fillna('S', inplace=True)
X_test['Embarked'].fillna('S', inplace=True)


X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)
X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)

print(X_train.info())
print(X_test.info())


dict_vec = DictVectorizer(sparse=False)
X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
X_test = dict_vec.transform(X_test.to_dict(orient='record'))

dict_vec.feature_names_


rfc = RandomForestClassifier()
cross_val_score(rfc, X_train, y_train, cv=5).mean()
# 0.8069863787583957

xgbc = XGBClassifier()
cross_val_score(xgbc, X_train, y_train, cv=5).mean()
# 0.8159500345238844

rfc.fit(X_train, y_train)
rfc_y_predict = rfc.predict(X_test)

rfc_submission = pd.DataFrame(
    {'PassengerId': test['PassengerId'], 'Survived': rfc_y_predict})

rfc_submission.to_csv(
    '/Users/hejinyang/Desktop/machine_learning/pratice/titanic_rfc_submission.csv', index=False)

xgbc.fit(X_train, y_train)
xgbc_y_predict = xgbc.predict(X_test)

xgbc_submission = pd.DataFrame(
    {'PassengerId': test['PassengerId'], 'Survived': xgbc_y_predict})

xgbc_submission.to_csv(
    '/Users/hejinyang/Desktop/machine_learning/pratice/titanic_xgb_submission.csv', index=False)


params = {'max_depth': range(2, 7), 'n_estimators': range(
    100, 1100, 200), 'learning_rate': [0.05, 0.1, 0.25, 0.5, 1.0], 'subsample': [0.5, 0.6, 0.7, 0.8]}

xgbc_best = XGBClassifier()

gs = GridSearchCV(xgbc_best, params, n_jobs=-1, cv=5, verbose=1)

gs.fit(X_train, y_train)


gs.best_score_
# 0.839539263071998
gs.best_params_
# {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 700, 'subsample': 0.5}

xgbc_best_y_predict = gs.predict(X_test)

xgbc_best_submission = pd.DataFrame(
    {'PassengerId': test['PassengerId'], 'Survived': xgbc_best_y_predict})

xgbc_best_submission.to_csv(
    '/Users/hejinyang/Desktop/machine_learning/pratice/titanic_xgb_best_submission.csv', index=False)
