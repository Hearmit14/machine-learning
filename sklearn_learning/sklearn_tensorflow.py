# tensorflow会话执行
import tensorflow as tf
import numpy as np

greeting = tf.constant("Hello FangFang")

# 启动一个会话
sess = tf.Session()

# 使用会话执行计算模块
result = sess.run(greeting)

# 输出会话执行结果
print(result)

# 关闭会话
sess.close()

# 使用tensorflow完成一次线性计算
# matrix1为1*2的行向量
matrix1 = tf.constant([[3., 3.]])
# matrix2为2*1的列向量
matrix2 = tf.constant([[2.], [2.]])
# 两个向量相乘
product = tf.matmul(matrix1, matrix2)
# 将乘积结果和一个标量拼接
linear = tf.add(product, tf.constant(2.0))
# 直接在会话中执行linear
with tf.Session() as sess:
    result = sess.run(linear)
    print(result)
