# 特征筛选：选择不同比例的特征进行测试，选择效果最好的特征。chi2是卡方检验。
# 使用特征筛选器；学会如何筛选出最适合的特征值
import pylab as pl
from sklearn.model_selection import cross_val_score
from sklearn import feature_selection
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

titanic = pd.read_csv(
    'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

y = titanic['survived']
X = titanic.drop(['row.names', 'name', 'survived'], axis=1)

X['age'].fillna(X['age'].mean(), inplace=True)
X.fillna('UNKNOWN', inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=33)

vec = DictVectorizer()
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

print(len(vec.feature_names_))

# 使用决策树进行预测
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train, y_train)
y_predict = dt.predict(X_test)
dt.score(X_test, y_test)

# fs返回最佳的前20%个特征 chi2是卡方检验 用来计算单一特征与类别之间的相关性
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=20)

X_fs_train = fs.fit_transform(X_train, y_train)
dt.fit(X_fs_train, y_train)

X_fs_test = fs.transform(X_test)
dt.score(X_fs_test, y_test)

# 通过交叉验证选择最合适的百分比
percentiles = np.arange(1, 100, 2)
results = []

for i in percentiles:
    fs = feature_selection.SelectPercentile(
        feature_selection.chi2, percentile=i)
    X_train_fs = fs.fit_transform(X_train, y_train)
    # cv选择每次测试的折数 按照5折 每次1折作为测试集 其余作为训练集 不断循环 每一折都做一次测试集
    scores = cross_val_score(dt, X_train_fs, y_train, cv=5)
    # 更新results 不断加入平均分数
    results = np.append(results, scores.mean())

print(results)


opt = np.where(results == results.max())[0]
print('Opt:', np.array(percentiles)[opt])

percentiles = percentiles.reshape(-1, 1)
results = results.reshape(-1, 1)

pl.plot(percentiles, results)
pl.xlabel('%%percentiles of features')
pl.ylabel('accuracy')
pl.show()

# 筛选特征后在测试集上进行评估
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=7)
X_train_fs = fs.fit_transform(X_train, y_train)
dt.fit(X_train_fs, y_train)

X_test_fs = fs.transform(X_test)
dt.score(X_test_fs, y_test)
