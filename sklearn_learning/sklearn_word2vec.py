# 使用词袋法对文本进行向量化
import re
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from gensim.models import word2vec
from bs4 import BeautifulSoup
from sklearn.datasets import fetch_20newsgroups
import nltk
from sklearn.feature_extraction.text import CountVectorizer
sent1 = 'The cat is walking in the bedroom.'
sent2 = 'A dog was running across the kitchen.'
count_vec = CountVectorizer()
sentences = [sent1, sent2]
print count_vec.fit_transform(sentences).toarray()
print count_vec.get_feature_names()
# 首先统计都出现了哪些词（少了‘A’）然后再建立与词数等长的向量 对应sent根据词频填入相应向量
# CountVectorizer的fit会过滤长度为1的停用词 例如 A
# 对于一个由字符串构成的数组，每个元素可能是一个以空格分割的句子（sentence），
# CountVectorizer.fit的功能是将它们分割，为每一个单词（word）编码，
# 在这个过程中会自动滤除停止词（stop words），例如英文句子中的”a”，”.”之类的长度为1的字符串。
# CountVectorizer.transform的功能则是将输入的数组中每个元素进行分割，然后使用fit中生成的编码字典，将原单词转化成编码，
# 使用NLTK进行语言学分析
nltk.download('punkt')
# 对句子进行分割
tokens_1 = nltk.word_tokenize(sent1)
print(tokens_1)
tokens_2 = nltk.word_tokenize(sent2)
print(tokens_2)
# 将词表按照Ascii排列
vocab_1 = sorted(set(tokens_1))
vocab_2 = sorted(set(tokens_2))
print(vocab_1)
print(vocab_2)
# 找到词汇原始的词根
stemmer = nltk.stem.PorterStemmer()
stem_1 = [stemmer.stem(t) for t in tokens_1]
stem_2 = [stemmer.stem(t) for t in tokens_2]
print(stem_1)
print(stem_2)
nltk.download('averaged_perceptron_tagger')
# 对词性进行标注
pos_tag_1 = nltk.tag.pos_tag(tokens_1)
pos_tag_2 = nltk.tag.pos_tag(tokens_2)
print(pos_tag_1)
print(pos_tag_2)


词向量分析word2vec: 每个连续词汇片段的最后一个单词都受前面单词的制约。
# 词向量训练
news = fetch_20newsgroups(subset='all')
X, y = news.data, news.target
# 把新闻中的句子剥离出来 并且形成列表


def news_to_sentences(news):
    news_text = BeautifulSoup(news).get_next()
    tokenizer = nltk.data.load('tokenizers/punk/english.pickle')
    raw_sentences = tokenizer.tokenize(news_text)
    sentences = []
    for sent in raw_sentences:
        sentences.append(
            re.sun('[^a-zA-Z]', ' ', sent.lower().strip()).split())
    return sentences


sentences = []
# 将长篇新闻中的句子剥离出来
#import news_to_sentences
for x in X:
    sentences = sentences+news_to_sentences(x)

num_features = 300
min_word_count = 20
num_workers = 2
context = 5
downsampling = le-3

model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features,
                          min_count=min_word_count, window=context, sample=downsampling)
model.init_sims(replaces=True)
model.most_similar('morning')
没有跑出结果，也没有理解问题（先放着，后面会专门写一篇填坑。）
xgboost: 自动利用cpu的多线程进行并行，全称是eXtreme Gradient Boosting.

# Xgboost模型
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

# 使用随机森林进行预测
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
print('RandomForestClassifier:', rfc.score(X_test, y_test))
# 使用Xgboost进行预测
xgbc = XGBClassifier()
xgbc.fit(X_train, y_train)
print('XGBClassifier:', xgbc.score(X_test, y_test))
Tensorflow: 利用会话来执行计算任务，初步进行了使用以及逐步进行了一个机器学习的基本任务。
# tensorflow会话执行
greeting = tf.constant("Hello FangFang")
# 启动一个会话
sess = tf.Session()
# 使用会话执行计算模块
result = sess.run(greeting)
print(result)
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
train = pd.read_csv('../python/Datasets/Breast-Cancer/breast-cancer-train.csv')
test = pd.read_csv('../python/Datasets/Breast-Cancer/breast-cancer-test.csv')
X_train = np.float32(train[['Clump Thickness', 'Cell Size']].T)
y_train = np.float32(train['Type'].T)
X_test = np.float32(test[['Clump Thickness', 'Cell Size']].T)
y_test = np.float32(test['Type'].T)
# 定义一个tensorflow的变量b作为截距
b = tf.Variable(tf.zeros([1]))
# 定义一个变量w作为线性模型的系数(-1.0在1.0之间的均匀分布)
w = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
# 显示定义线性函数
y = tf.matmul(w, X_train)+b
# 使用reduce_mean获得均方误差
loss = tf.reduce_mean(tf.square(y-y_train))
# 使用梯队下降估计w,b,并设计迭代步长
optimizer = tf.train.GradientDescentOptimizer(0.01)
# 以最小二乘为优化目标
train = optimizer.minimize(loss)
# 初始化所有变量
init = tf.initialize_all_variables()
# 开启会话
sess = tf.Session()
# 执行初始化变量操作
sess.run(init)
# 迭代1000次 训练参数
for step in range(0, 1000):
    sess.run(train)
    if step % 200 == 0:
        print(step, sess.run(w), sess.run(b))
test_negative = test.loc[test['Type'] == 0][['Clump Thickness', 'Cell Size']]
test_postive = test.loc[test['Type'] == 0][['Clump Thickness', 'Cell Size']]
plt.scatter(test_negative['Clump Thickness'],
            test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(test_postive['Clump Thickness'],
            test_postive['Cell Size'], marker='x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
lx = np.arange(0, 12)
# 以0.5为界 1为恶 0为良
ly = (0.5-sess.run(b)-lx*sess.run(w)[0][0])/sess.run(w)[0][1]
plt.plot(lx, ly, color='blue')
plt.show()
