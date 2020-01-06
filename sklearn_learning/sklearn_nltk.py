# 使用词袋法对文本进行向量化
import nltk
from sklearn.feature_extraction.text import CountVectorizer

sent1 = 'The cat is walking in the bedroom.'
sent2 = 'A dog was running across the kitchen.'

count_vec = CountVectorizer()
sentences = [sent1, sent2]

print(count_vec.fit_transform(sentences).toarray())
print(count_vec.get_feature_names())

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
