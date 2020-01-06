# 超参数搜索方法：网格搜索：单线程以及并行搜索。
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.preprocessing import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

news = fetch_20newsgroups(subset='all')

X_train, X_test, y_train, y_test = train_test_split(
    news.data[:3000], news.target[:3000], test_size=0.25, random_state=33)

# pipeline 一种简化代码的方法 先数据处理再预测
clf = Pipeline([('vect', TfidfVectorizer(
    stop_words='english', analyzer='word')), ('svc', SVC())])

parameters = {
    'svc__gamma': np.logspace(-2, 1, 4), 'svc__C': np.logspace(-1, 1, 3)}

# 单线程
gs = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3)
# 多线程，使用全部cpu
gs = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3, n_jobs=-1)

gs.fit(X_train, y_train)

gs.best_params_
print(gs.score(X_test, y_test))
