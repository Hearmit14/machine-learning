import numpy as np
test1 = np.array([[5, 10, 15],
                  [20, 25, 30],
                  [35, 40, 45]])

test1.sum()
# 输出 225
test1.max()
# 输出 45
test1.min()
# 输出 5
test1.mean()
# 输出 25.0

test1.sum(axis=1)
# array([30,  75, 120])
test1.sum(axis=0)
# array([60, 75, 90])

test2 = np.array([[2, 3, 5],
                  [3, 5, 3],
                  [5, 4, 5]])

# 对应位置元素相乘
print(test1 * test2)
# [[10  30  75]
#  [60 125  90]
#  [175 160 225]]
# 矩阵乘法
print(test1.dot(test2))
# [[115 125 130]
#  [265 305 325]
#  [415 485 520]]
# 矩阵乘法，同上
print(np.dot(test1, test2))
# [[115 125 130]
#  [265 305 325]
#  [415 485 520]]

a = np.array(range(4))

print(a)
print(a**2)
print(np.exp(a))
print(np.sqrt(a))

# np.random.random((3, 4))

test = np.floor(10 * np.random.random((3, 4)))
test.T
test.shape = (6, 2)
test.shape = (2, -1)
test3 = test.reshape((-1, 2))

np.arange(10, 30, 5)
np.arange(12).reshape(3, 4)
