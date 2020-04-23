from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from gensim.models import word2vec
import nltk.data
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
import pandas as pd

train = pd.read_csv(
    '/Users/hejinyang/Desktop/machine_learning/pratice/IMDB/labeledTrainData.tsv', delimiter='\t')
test = pd.read_csv(
    '/Users/hejinyang/Desktop/machine_learning/pratice/IMDB/testData.tsv', delimiter='\t')

train.head()
test.head()


def review_to_text(review, remove_stopwords):
    # 去除html标记
    raw_text = BeautifulSoup(review, 'html').get_text()
    # 去除非字母
    letters = re.sub('[^a-zA-Z]', ' ', raw_text)
    words = letters.lower().split()
    # 去除停用词
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
    return words


X_train = []
for review in train['review']:
    X_train.append(' '.join(review_to_text(review, True)))

y_train = train['sentiment']

X_test = []
for review in test['review']:
    X_test.append(' '.join(review_to_text(review, True)))

# 分别使用CountVectorizer与TfidfVectorizer提取文本特征
pip_count = Pipeline(
    [('count_vec', CountVectorizer(analyzer='word')), ('mnb', MultinomialNB())])

pip_tfidf = Pipeline(
    [('tfidf_vec', TfidfVectorizer(analyzer='word')), ('mnb', MultinomialNB())])

params_count = {'count_vec__binary': [True, False], 'count_vec__ngram_range': [
    (1, 1), (1, 2)], 'mnb__alpha': [0.1, 1.0, 10.0]}
params_tfidf = {'tfidf_vec__binary': [True, False], 'tfidf_vec__ngram_range': [
    (1, 1), (1, 2)], 'mnb__alpha': [0.1, 1.0, 10.0]}

gs_count = GridSearchCV(pip_count, params_count, cv=4, n_jobs=-1, verbose=1)
gs_tfidf = GridSearchCV(pip_tfidf, params_tfidf, cv=4, n_jobs=-1, verbose=1)

gs_count.fit(X_train, y_train)
gs_count.best_score_
# 0.88216
gs_count.best_params_
# {'count_vec__binary': True, 'count_vec__ngram_range': (
#     1, 2), 'mnb__alpha': 1.0}
count_y_predict = gs_count.predict(X_test)


gs_tfidf.fit(X_train, y_train)
gs_tfidf.best_score_
# 0.88712
gs_tfidf.best_params_
# {'mnb__alpha': 0.1, 'tfidf_vec__binary': True,
#     'tfidf_vec__ngram_range': (1, 2)}
tfidf_y_predict = gs_tfidf.predict(X_test)

submission_count = pd.DataFrame(
    {'id': test['id'], 'sentiment': count_y_predict})

submission_tfidf = pd.DataFrame(
    {'id': test['id'], 'sentiment': tfidf_y_predict})


submission_count.to_csv(
    '/Users/hejinyang/Desktop/machine_learning/pratice/IMDB/imdb_count_submission.csv', index=False)
submission_tfidf.to_csv(
    '/Users/hejinyang/Desktop/machine_learning/pratice/IMDB/imdb_tfidf_submission.csv', index=False)


# 词向量分析word2vec
unlabeled_train = pd.read_csv(
    '/Users/hejinyang/Desktop/machine_learning/pratice/IMDB/unlabeledTrainData.tsv', delimiter='\t', quoting=3)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def review_to_sentences(review, tokenizer):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_text(raw_sentence, False))
    return sentences


corpora = []
for review in unlabeled_train['review']:
    corpora += review_to_sentences(
        review.encode('utf-8').decode('utf-8'), tokenizer)

# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 20   # Minimum word count
num_workers = 12       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

model = word2vec.Word2Vec(corpora, workers=num_workers,
                          size=num_features, min_count=min_word_count,
                          window=context, sample=downsampling, sg=1)

model.init_sims(replace=True)

# 保存训练好的词向量模型
model_name = "/Users/hejinyang/Desktop/machine_learning/pratice/IMDB/300features_20minwords_10context"
model.save(model_name)

model = word2vec.Word2Vec.load(
    "/Users/hejinyang/Desktop/machine_learning/pratice/IMDB/300features_20minwords_10context")

model.most_similar("man")
model.most_similar("news")
model.most_similar("film")

model['man']
model.wv['man']


def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0.
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model.wv[word])
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    for review in reviews:
        reviewFeatureVecs[counter] = makeFeatureVec(
            review, model, num_features)
        counter += 1
    return reviewFeatureVecs


clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(review_to_text(review, remove_stopwords=True))

# aaa = makeFeatureVec(clean_train_reviews[0], model, num_features)

trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)
trainDataVecs[0]
trainDataVecs.shape

clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(review_to_text(review, remove_stopwords=True))

testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)
testDataVecs.shape


gbc = GradientBoostingClassifier()
params_gbc = {'n_estimators': [100, 500, 700], 'learning_rate': [
    0.01, 0.1, 1.0], 'max_depth': [2, 3, 4]}

gs = GridSearchCV(gbc, params_gbc, cv=4, n_jobs=-1, verbose=1)

gs.fit(trainDataVecs, y_train)
# [Parallel(n_jobs=-1)]: Done 108 out of 108 | elapsed: 179.8min finished
gs.best_score_
# 0.8588
gs.best_params_
# {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 700}
result_gbc = gs.predict(testDataVecs)

# Write the test results
output_gbc = pd.DataFrame(data={"id": test["id"], "sentiment": result_gbc})
output_gbc.to_csv("/Users/hejinyang/Desktop/machine_learning/pratice/IMDB/imdb_w2v_submission.csv",
                  index=False, quoting=3)


xgbc = XGBClassifier()
params_xgbc = {'n_estimators': [100, 500, 700], 'learning_rate': [
    0.01, 0.1, 1.0], 'max_depth': [2, 3, 4]}

gs = GridSearchCV(xgbc, params_xgbc, cv=4, n_jobs=-1, verbose=1)
# [Parallel(n_jobs=-1)]: Done 108 out of 108 | elapsed: 50.1min finished

gs.fit(trainDataVecs, y_train)

gs.best_score_
# 0.85916
gs.best_params_
# {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 700}
result_xgbc = gs.predict(testDataVecs)

output_xgbc = pd.DataFrame(data={"id": test["id"], "sentiment": result_xgbc})
output_xgbc.to_csv("/Users/hejinyang/Desktop/machine_learning/pratice/IMDB/imdb_w2v_xgbc_submission.csv",
                   index=False, quoting=3)
