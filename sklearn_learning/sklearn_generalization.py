from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# 比萨直径与售价的关系
X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# linspace均匀采样在0,-26之间采样100个点
xx = np.linspace(0, 26, 100)
xx = xx.reshape(xx.shape[0], 1)
yy = regressor.predict(xx)

plt.scatter(X_train, y_train)

# 设置legend图例：一次拟合 直线
plt1, = plt.plot(xx, yy, label="Degree=1")
plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
# 为完全控制，将句柄传递给legend
plt.legend(handles=[plt1])
plt.show()
print('regressor :', regressor.score(X_train, y_train))


# 使用二次多项式模型
# y=a+bx+cx^2
poly2 = PolynomialFeatures()
X_train_poly2 = poly2.fit_transform(X_train)

# fit_transform之后[6]变成了[1,6,36]
regressor_poly2 = LinearRegression()
regressor_poly2.fit(X_train_poly2, y_train)

xx_poly2 = poly2.transform(xx)
yy_poly2 = regressor_poly2.predict(xx_poly2)

plt.scatter(X_train, y_train)
plt1, = plt.plot(xx, yy, label='Degree=1')
plt2, = plt.plot(xx, yy_poly2, label='Degree=2')
# 设置横纵坐标轴
plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')

# 为完全控制，将句柄传递给legend
plt.legend(handles=[plt1, plt2])
plt.show()
print('regressor_poly :', regressor_poly2.score(X_train_poly2, y_train))


# 使用四次多项式模型
# y=a+bx+cx^2
poly4 = PolynomialFeatures(degree=4)
X_train_poly4 = poly4.fit_transform(X_train)
regressor_poly4 = LinearRegression()
regressor_poly4.fit(X_train_poly4, y_train)
xx_poly4 = poly4.transform(xx)
yy_poly4 = regressor_poly4.predict(xx_poly4)
plt.scatter(X_train, y_train)
plt1, = plt.plot(xx, yy, label='Degree=1')
plt2, = plt.plot(xx, yy_poly2, label='Degree=2')
plt4, = plt.plot(xx, yy_poly4, label='Degree=4')

# 设置横纵坐标轴
plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
# 为完全控制，将句柄传递给legend
plt.legend(handles=[plt1, plt2, plt4])
plt.show()
print('regressor_poly :', regressor_poly4.score(X_train_poly4, y_train))

# 测试集进行测试
X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]
regressor.score(X_test, y_test)
X_test_poly2 = poly2.transform(X_test)
X_test_poly4 = poly4.transform(X_test)
regressor_poly2.score(X_test_poly2, y_test)
regressor_poly4.score(X_test_poly4, y_test)


#加入L1正则 :Lasso
lasso_poly4 = Lasso()
lasso_poly4.fit(X_train_poly4, y_train)
print(lasso_poly4.score(X_test_poly4, y_test))
# coef输出函数的参数
print(lasso_poly4.coef_)
print(regressor_poly4.coef_)

#加入L2正则 :Ridge
ridge_poly4 = Ridge()
ridge_poly4.fit(X_train_poly4, y_train)

print(ridge_poly4.score(X_test_poly4, y_test))
print(ridge_poly4.coef_)
print(np.sum(lasso_poly4.coef_**2))
print(np.sum(ridge_poly4.coef_**2))
print(np.sum(regressor_poly4.coef_ ** 2))
