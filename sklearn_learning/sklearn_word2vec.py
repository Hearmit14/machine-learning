# 词向量分析word2vec:每个连续词汇片段的最后一个单词都受前面单词的制约。
# 词向量训练
import re
import nltk
from gensim.models import word2vec
from bs4 import BeautifulSoup
from sklearn.datasets import fetch_20newsgroups

news = fetch_20newsgroups(subset='all')
X, y = news.data, news.target


# get_text()改变数据类型
# 把新闻中的句子剥离出来 并且形成列表
def news_to_sentences(news):
    news_text = BeautifulSoup(news).get_text()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(news_text)
    sentences = []
    for sent in raw_sentences:
        sentences.append(
            re.sub('[^a-zA-Z]', ' ', sent.lower().strip()).split())
    return sentences


sentences = []
# 将长篇新闻中的句子剥离出来
#import news_to_sentences
for x in X:
    sentences = sentences+news_to_sentences(x)

# 词向量维度
num_features = 300
# 被考虑词汇频度
min_word_count = 20
# cpu核心数
num_workers = 2
# 窗口大小
context = 5
downsampling = 1e-3

model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features,
                          min_count=min_word_count, window=context, sample=downsampling)

model.init_sims(replace=True)
model.most_similar('morning')
model.most_similar('email')
model.most_similar('news')
