from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer,load_diabetes


data = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], train_size=0.8, random_state=0)


regressor = DecisionTreeClassifier(random_state=0)
parameters = {'max_depth': range(1, 6)}
scoring_fnc = make_scorer(accuracy_score)
kfold = KFold(n_splits=10)


grid = GridSearchCV(regressor, parameters, scoring_fnc, cv=kfold)
grid = grid.fit(X_train, y_train)
reg = grid.best_estimator_

print('best score: %f'%grid.best_score_)

print('best parameters:')
for key in parameters.keys():
    print('%s: %d'%(key, reg.get_params()[key]))


print('test score: %f'%reg.score(X_test, y_test))

import pandas as pd
import numpy as np
pd.DataFrame(grid.cv_results_).T


################################################################################################################################################################
# 跑一个弹性网络
from sklearn.linear_model import ElasticNetCV
from sklearn.datasets import make_regression
 
diabetes = load_diabetes()
x1 = diabetes.data
y1 = diabetes.target

x2, y2 = make_regression(n_features=2, random_state=0)

regr = ElasticNetCV(l1_ratio=np.logspace(-2, 1, 20),alphas=np.logspace(-3, 2, 50))
regr.fit(X_train, y_train)
regr.alpha_
regr.l1_ratio_
regr.score(X_test,y_test)









param_test1 ={'l1_ratio': [.01, .1, .5, .9, .99, 1],'alphas': [0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10]}  

gsearch1= GridSearchCV(estimator=ElasticNetCV(),param_grid=param_test1,scoring='roc_auc',cv=5)


gsearch1.fit(X_train,y_train)

gsearch1.fit(x1,y1)

gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_  

