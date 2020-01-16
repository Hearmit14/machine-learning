import matplotlib.pyplot as plt
import pandas as pd
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


s = pd.Series([1, 3, 4, np.nan, 7, 9])
t = pd.Series(s, index=[1, 1, 2, 2, 'a', 4])

# 新建对象 Object Creation
# 通过传入一个numpy的数组、指定一个时间的索引以及一个列名。
dates = pd.date_range('20190101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
df2 = pd.DataFrame(np.arange(12).reshape(3, 4))


# 查看 Viewing Data
df.head()
df.tail(3)
df.index
df.columns
df2.columns

df.describe()
df.info()

df.sort_index(axis=1)
df.sort_index(axis=1, ascending=False)
df.sort_values(by='B')
df.sort_values(by='B', ascending=False)


# 筛选 Selection
# 获取某列
df['A']
# 选择多行
df[0:3]

# 选择某行,通过标签选择
df.loc[dates[0]]
# 选择指定行列的数据
df.loc[:, ('A', 'C')]
df.loc['20190102':'20190105', ('A', 'C')]

# 选择某行,通过位置选择
df.iloc[3]
# 选择指定行列的数据
df.iloc[3:5, 0:2]

# 按条件判断选择
df[df.A > 0]
df[df > 0]


# 空值处理 Missing Data
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1], 'E'] = 1
df1.dropna(how='any')
df1.fillna(value=5)


# 运算 Operations
# 注意 所有的统计默认是不包含空值的
df.mean()

# 这里将s的值移动两个，那么空出的部分会自动使用NaN填充
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates)
s = s.shift(2)

# 通过apply()方法，可以对数据进行逐一操作
# 累计求和
df.apply(np.cumsum)
# 自定义方法
df.apply(lambda x: x.max() - x.min())

df['A'].value_counts()


# String方法
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s.str.upper()

s = pd.Series(['A,b', 'c,d'])
s.str.split(',', expand=True)


# 合并 Merge
df = pd.DataFrame(np.random.randn(10, 4))

# Concat方法
pieces = [df[:3], df[3:7], df[7:]]
pd.concat(pieces)

# Merge方法
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
pd.merge(left, right, on='key')

# Append方法
df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])
s = df.iloc[3]
df.append(s, ignore_index=True)


# 分组 Grouping
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
                         'foo', 'bar', 'foo', 'foo'],
                   'B': ['one', 'one', 'two', 'three',
                         'two', 'two', 'one', 'three'],
                   'C': np.random.randn(8),
                   'D': np.random.randn(8)})

df.groupby('A').sum()
df.groupby(['A', 'B']).sum()


# 整形 Reshaping
# 堆叠 Stack,python的zip函数可以将对象中对应的元素打包成一个个的元组
tuples = list(zip(['bar', 'bar', 'baz', 'baz',
                   'foo', 'foo', 'qux', 'qux'],
                  ['one', 'two', 'one', 'two',
                   'one', 'two', 'one', 'two']))

tuples1 = np.array([['bar', 'bar', 'baz', 'baz',
                     'foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two',
                     'one', 'two', 'one', 'two']])

index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])

# 分类目录类型 Categoricals
# 类型转换：astype('category')
df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                   "raw_grade": ['a', 'b', 'b', 'a', 'a', 'e']})

df['grade'] = df['raw_grade'].astype('category')
df['grade']
# 重命名分类：cat
df["grade"].cat.categories = ["very good", "good", "very bad"]
df['grade']
# 重分类：
df['grade'] = df['grade'].cat.set_categories(
    ["very bad", "bad", "medium", "good", "very good"])
df['grade']
# 排列
df.sort_values(by="grade")
# 分组
df.groupby("grade").size()

# 画图 Plotting
# Series画图
ts = pd.Series(np.random.randn(1000),
               index=pd.date_range('1/1/2019', periods=1000))
ts = ts.cumsum()

ts.plot()
plt.show()

# DataFrame画图
df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,
                  columns=['A', 'B', 'C', 'D'])
df = df.cumsum()
plt.figure()
df.plot()
plt.legend(loc='best')
plt.show()


# 导入导出数据 Getting Data In/Out
df.to_csv('foo.csv')
pd.read_csv('foo.csv')
