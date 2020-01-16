import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def gradient(x):
    return np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)])


def hessian(x):
    return np.array([[-400*(x[1]-3*x[0]**2)+2, -400*x[0]], [-400*x[0], 200]])


def newton(x):
    print("初始点为 : ", x)
    res = []
    res.append(x)
    i = 1
    imax = 1000
    delta = 1
# 迭代的条件是小于imax，或者是更新的距离小于一个很小的数值
    # while i < imax and delta > 10**(-5):
    #     p = -np.dot(np.linalg.inv(hessian(x)), gradient(x))
    #     x_new = x + p
    #     res.append(x_new)
    #     delta = sum((x-x_new)**2)   # 更新的距离
    #     print("初始点为 : ", x_new)
    #     i = i+1
    #     x = x_new  # 更新x
    # return np.array(res)
    while i < imax and delta > 10**(-5):
        x = x - np.dot(np.linalg.inv(hessian(x)), gradient(x))
        res.append(x)
        i = i+1
        delta = sum(p**2)   # 更新的距离,所有的元素的累积平方和
    return np.array(res)


def gradient_descent(x):
    print("初始点为 : ", x)
    res = []
    res.append(x)
    i = 1
    imax = 200000
    delta = 1
    alpha = 0.001
    while i < imax and delta > 10 ** (-5):
        x = x - alpha*gradient(x)
        res.append(x)
        i = i + 1
        delta = sum(alpha*gradient(x)**2)   # 更新的距离,所有的元素的累积平方和
    return np.array(res)


x = gradient_descent([2, 1.5])
x


if __name__ == "__main__":
    X1 = np.arange(-3, 3+0.05, 0.05)
    X2 = np.arange(-3, 3+0.05, 0.05)
    [x1, x2] = np.meshgrid(X1, X2)
    plt.contour(x1, x2, f([x1, x2]), 100)  # 画出函数的20条轮廓线
    x0 = np.array([2, 2])
    res1 = newton(x0)
    res2 = gradient_descent(x0)
    res_x1 = res1[:, 0]
    res_y1 = res1[:, 1]
    plt.plot(res_x1, res_y1)
    res_x2 = res2[:, 0]
    res_y2 = res2[:, 1]
    plt.plot(res_x2, res_y2)
    plt.show()
