from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
data = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', names=column_names)

# 将?替换为标准缺失值表示。
data = data.replace(to_replace='?', value=np.nan)
# 丢弃带有缺失值的数据（只要有一个维度有缺失）。
data = data.dropna(how='any')

# 输出data的数据量和维度。
data.shape

data.head(3)


X_train, X_test, y_train, y_test = train_test_split(
    data[column_names[1:10]], data[column_names[10]], test_size=0.25, random_state=33)

y_train.value_counts()

y_test.value_counts()

# 从sklearn.preprocessing里导入StandardScaler。
# 从sklearn.linear_model里导入LogisticRegression与SGDClassifier。

# 标准化数据，保证每个维度的特征数据方差为1，均值为0。使得预测结果不会被某些维度过大的特征值而主导。
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

lr = LogisticRegression()
sgdc = SGDClassifier()

# 调用LogisticRegression中的fit函数/模块用来训练模型参数。
lr.fit(X_train, y_train)
# 使用训练好的模型lr对X_test进行预测，结果储存在变量lr_y_predict中。
lr_y_predict = lr.predict(X_test)

# 调用SGDClassifier中的fit函数/模块用来训练模型参数。
sgdc.fit(X_train, y_train)
# 使用训练好的模型sgdc对X_test进行预测，结果储存在变量sgdc_y_predict中。
sgdc_y_predict = sgdc.predict(X_test)

# 从sklearn.metrics里导入classification_report模块。

# 使用逻辑斯蒂回归模型自带的评分函数score获得模型在测试集上的准确性结果。
print('Accuracy of LR Classifier:', lr.score(X_test, y_test))
# 利用classification_report模块获得LogisticRegression其他三个指标的结果。
print(classification_report(y_test, lr_y_predict,
                            target_names=['Benign', 'Malignant']))

# 使用随机梯度下降模型自带的评分函数score获得模型在测试集上的准确性结果。
print('Accuarcy of SGD Classifier:', sgdc.score(X_test, y_test))
# 利用classification_report模块获得SGDClassifier其他三个指标的结果。
print(classification_report(y_test, sgdc_y_predict,
                            target_names=['Benign', 'Malignant']))
