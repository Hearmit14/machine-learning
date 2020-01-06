# 特征抽取：对特征进行向量化：根据词频；根据词频和文档频率；以及是否考虑停用词
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import DictVectorizer

# 向量化
measurements = [{'city': 'Dubai', 'temperature': '33.'}, {'city': 'London',
                                                          'temperature': '12.'}, {'city': 'San Fransisco', 'temperature': '18.'}]

vec = DictVectorizer()
# DictVectorizer对特征进行抽取和细化:将dict类型的list数据，转换成numpy array
vec.fit_transform(measurements).toarray()
vec.get_feature_names()

# 不去掉停用词进行对比
# 使用CountVectorizer(只根据词频)进行向量化
news = fetch_20newsgroups(subset='all')

X_train, X_test, y_train, y_test = train_test_split(
    news.data, news.target, test_size=0.25, random_state=33)

count_vec = CountVectorizer()

# 只统计词频 默认不去除停用词
X_count_train = count_vec.fit_transform(X_train)
X_count_test = count_vec.transform(X_test)

mnb_count = MultinomialNB()
mnb_count.fit(X_count_train, y_train)
mnb_count_y_predict = mnb_count.predict(X_count_test)

print('the accuracy :', mnb_count.score(X_count_test, y_test))
print(classification_report(
    y_test, mnb_count_y_predict, target_names=news.target_names))


# 使用TfidfVectorizer(根据词频和文档频率)进行向量化
news = fetch_20newsgroups(subset='all')

X_train, X_test, y_train, y_test = train_test_split(
    news.data, news.target, test_size=0.25, random_state=33)

tfi_vec = TfidfVectorizer()

# 统计词频以及文档频率 默认不去除停用词
X_tfi_train = tfi_vec.fit_transform(X_train)
X_tfi_test = tfi_vec.transform(X_test)


mnb_tfi = MultinomialNB()
mnb_tfi.fit(X_tfi_train, y_train)
mnb_tfi_y_predict = mnb_tfi.predict(X_tfi_test)

print('the accuracy :', mnb_tfi.score(X_tfi_test, y_test))
print(classification_report(
    y_test, mnb_tfi_y_predict, target_names=news.target_names))


# 使用停用词进行对比
# 设置停用词为‘english’则表示调用系统默认的英文停用词
count_filter_vec, tfi_filter_vec = CountVectorizer(
    analyzer='word', stop_words='english'), TfidfVectorizer(analyzer='word', stop_words='english')

# 使用有停用词的CountVectorizer
X_count_filter_train = count_filter_vec.fit_transform(X_train)
X_count_filter_test = count_filter_vec.transform(X_test)

# 使用有停用词的TfidfVectorizer
X_tfi_filter_train = tfi_filter_vec.fit_transform(X_train)
X_tfi_filter_test = tfi_filter_vec.transform(X_test)

mnb_count_filter = MultinomialNB()
mnb_count_filter.fit(X_count_filter_train, y_train)
y_count_predict = mnb_count_filter.predict(X_count_filter_test)

mnb_tfi_filter = MultinomialNB()
mnb_tfi_filter.fit(X_tfi_filter_train, y_train)
y_tfi_predict = mnb_tfi_filter.predict(X_tfi_filter_test)

print('the accuracy :', mnb_count_filter.score(X_count_filter_test, y_test))
print(classification_report(y_test, y_count_predict, target_names=news.target_names))

print('the accuracy :', mnb_tfi_filter.score(X_tfi_filter_test, y_test))
print(classification_report(y_test, y_tfi_predict, target_names=news.target_names))
