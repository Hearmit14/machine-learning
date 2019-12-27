# 1、创建RDD
# 从文件系统中加载数据创建RDD
# 从本地加载文件
# local方式启动

# 任何Spark程序都是SparkContext开始的，SparkContext的初始化需要一个SparkConf对象，SparkConf包含了Spark集群配置的各种参数。

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Word Count") \
    .getOrCreate()

sc = spark.sparkContext


textFile = sc.textFile('file:////home/dev/hejinyang/word.txt')
textFile.first()
# 事先不需先建test目录
textFile.saveAsTextFile("file:///home/dev/hejinyang/test")

# 从hdfs文件系统加载文件
# yan方式启动
# hdfs dfs -ls /user/hive/warehouse/dw_stage_open/dw_stage_open.db/hejy_push/

textFile = sc.textFile("/user/hive/warehouse/dw_stage_open/dw_stage_open.db/hejy_push/zhimaq_4_20190103_08_37_28.csv")
textFile.first()


# 通过并行集合（数组）创建RDD
# 可以调用SparkContext的parallelize方法，在Driver中一个已经存在的集合（数组）上创建。
nums = [1,2,3,4,5]
rdd = sc.parallelize(nums)



#构造多维RDD
# 方法1，先构造多维dataframe
val df1 = Seq(
     |       (1.0, 2.0, 3.0),
     |       (1.1, 2.1, 3.1),
     |       (1.2, 2.2, 3.2)).toDF("c1", "c2", "c3")

df1.show
+---+---+---+
| c1| c2| c3|
+---+---+---+
|1.0|2.0|3.0|
|1.1|2.1|3.1|
|1.2|2.2|3.2|
+---+---+---+
                         
// DataFrame转换成RDD[Vector]
val rowsVector= df1.rdd.map {
            x =>
              Vectors.dense(
                x(0).toString().toDouble,
                x(1).toString().toDouble,
                x(2).toString().toDouble)
          }

# 方法1，构造多维数组
val rdd1= sc.parallelize(
  Array(
    Array(1.0,7.0,0,0),
    Array(0,2.0,8.0,0),
    Array(5.0,0,3.0,9.0),
    Array(0,6.0,0,4.0)
  )
).map(f => Vectors.dense(f))

# map( ):接收一个函数，应用到RDD中的每个元素，然后为每一条输入返回一个对象。
# flatMap( )：接收一个函数，应用到RDD中的每个元素，返回一个包含可迭代的类型(如list等)的RDD,可以理解为先Map()，后flat().

# 2、RDD操作
lines = sc.textFile('file:////home/dev/hejinyang/word.txt')
lineLengths = lines.map(lambda s : len(s))
totalLength = lineLengths.reduce(lambda a, b : a + b)
totalLength


lines.filter(lambda line : "b" in line).count()

# 单词个数
lines.map(lambda line : len(line.split(" "))).reduce(lambda a,b : (a > b and a or b))
# 每行长度
lines.map(lambda line : len(line)).reduce(lambda a,b : (a > b and a or b))

list = ["Hadoop","Spark","Hive"]
rdd = sc.parallelize(list)
print(rdd.count()) 
print(','.join(rdd.collect())) 

# 持久化
# 会调用persist(MEMORY_ONLY)，但是，语句执行到这里，并不会缓存rdd，这是rdd还没有被计算生成
rdd.cache()  


pairRDD = lines.flatMap(lambda line : line.split(" ")).map(lambda word : (word,1))


# 3、键值对RDD
# “键值对”是一种比较常见的RDD元素类型，分组和聚合操作中经常会用到。Spark操作中经常会用到“键值对RDD”（Pair RDD），用于完成聚合计算。
# 键值对RDD的创建

1
lines = sc.textFile('file:////home/dev/hejinyang/word.txt')
pairRDD = lines.flatMap(lambda line : line.split(" ")).map(lambda word : (word,1))

2
list = ["Hadoop","Spark","Hive","Spark"]
rdd = sc.parallelize(list)
pairRDD = rdd.map(lambda word : (word,1))
# pairRDD.foreach(print)
pairRDD.first()
print(pairRDD.collect()) 

# 常用的键值对转换操作
# reduceByKey(func)的功能是，使用func函数合并具有相同键的值。
print(pairRDD.reduceByKey(lambda a,b : a+b).collect())

# groupByKey()的功能是，对具有相同键的值进行分组。
print(pairRDD.groupByKey().collect())

# keys()只会把键值对RDD中的key返回形成一个新的RDD。
print(pairRDD.keys().collect())

# values()只会把键值对RDD中的value返回形成一个新的RDD。
print(pairRDD.values().collect())

# sortByKey()的功能是返回一个根据键排序的RDD
print(pairRDD.sortByKey().collect())

# mapValues(func)对键值对RDD中的每个value都应用一个函数，但key不会发生变化
print(pairRDD.mapValues(lambda x : x+1).collect())

# join 对于给定的两个输入数据集(K,V1)和(K,V2)，只有在两个数据集中都存在的key才会被输出，最终得到一个(K,(V1,V2))类型的数据集。
pairRDD1 = sc.parallelize([('spark','fast')])

print(pairRDD.join(pairRDD1).collect())


# 题目：给定一组键值对(“spark”,2),(“hadoop”,6),(“hadoop”,4),(“spark”,6)，键值对的key表示图书名称，value表示某天图书销量，请计算每个键对应的平均值，也就是计算每种图书的每天平均销量。
pairRDD2 = sc.parallelize([("spark",2),("hadoop",6),("hadoop",4),("spark",6)])

# 方法1
# 求出次数
pairRDD3 = pairRDD2.keys().map(lambda x : (x,1)).reduceByKey(lambda a,b : a+b)
# 求出总数
pairRDD4 = pairRDD2.reduceByKey(lambda a,b : a+b)
# 期初平均数
pairRDD5 = pairRDD3.join(pairRDD4).mapValues(lambda x:x[1]/x[0])

# 方法2
pairRDD4.mapValues(lambda x :(x,1)).reduceByKey(lambda x,y : (x[0]+y[0],x[1]+y[1])).mapValues(lambda x : (x[0]/x[1]))


# print(pairRDD2.keys().map(lambda x : (x,1)).reduceByKey(lambda a,b : a+b).collect())

# print(pairRDD2.reduceByKey(lambda a,b : a+b).collect())

# print(pairRDD2.mapValues(lambda x :(x,1)).reduceByKey(lambda x,y : x[0]+y[0],x[1]+y[1]).mapValues(lambda x : x[0]/x[1]).collect())

# print(pairRDD2.mapValues(lambda x :(x,1)).reduceByKey(lambda x,y : (x[0]+y[0],x[1]+y[1])).mapValues(lambda x : (x[0]/x[1])).collect())








# 1.DataFrame的创建
# 1
spark=SparkSession.builder.getOrCreate()
df = spark.read.json("file:///app/hadoop/spark/examples/src/main/resources/people.json")
df.show()

# 2
training = spark.createDataFrame([
    (0, "a b c d e spark", 1.0),
    (1, "b d", 0.0),
    (2, "spark f g h", 1.0),
    (3, "hadoop mapreduce", 0.0)
], ["id", "text", "label"])



# 打印模式信息
df.printSchema()

# 选择多列
df.select(df.name,df.age + 1).show()

# 条件过滤
df.filter(df.age > 20 ).show()
 
# 分组聚合
df.groupBy("age").count().show()

# 排序
df.sort(df.age.desc()).show()
 
#多列排序
df.sort(df.age.desc(), df.name.asc()).show()
 
#对列进行重命名
df.select(df.name.alias("username"),df.age).show()


# 2.从RDD转换得到DataFrame
# txt文件没有固定格式，只能用这种方法
from pyspark.sql.types import Row
def f(x):
    rel = {}
    rel['name'] = x[0]
    rel['age'] = x[1]
    return rel

peopleDF = sc.parallelize([("Justin",19),("Michael",29),("Andy",30)]).map(lambda line : line.split(',')).map(lambda x: Row(**f(x))).toDF()
peopleDF = sc.textFile("file:///usr/local/spark/examples/src/main/resources/people.txt").map(lambda line : line.split(',')).map(lambda x: Row(**f(x))).toDF()

peopleDF.createOrReplaceTempView("people")
personsDF = spark.sql("select * from people")
personsDF.rdd.map(lambda t : "Name:"+t[0]+","+"Age:"+t[1]).foreach(print)





# 连接Hive读写数据（DataFrame）

# drop table if exists sdk_user.hejy_temp_2;
# create table sdk_user.hejy_temp_2 as 
# select * from sdk_user.hejy_temp_1
# limit 10;


# 用hive数据创建dataframe
from pyspark.sql import HiveContext
hive_context = HiveContext(sc)
hive_context.sql('use sdk_user')
hive_context.sql('select * from sdk_user.hejy_temp_1 limit 2').show()

hiveDataFrame=hive_context.sql('select * from sdk_user.hejy_temp_1 limit 2')

hiveDataFrame.show()
hiveDataFrame.select(hiveDataFrame.dhid,hiveDataFrame.app_3 + 1).show()


# 用dataframe数据修改hive表
# from pyspark.sql.types import Row
# from pyspark.sql.types import StructType
# from pyspark.sql.types import StructField
# from pyspark.sql.types import StringType
# from pyspark.sql.types import IntegerType
# from pyspark.sql import HiveContext
# hive_context = HiveContext(sc)
# hive_context.sql('use sdk_user')
# studentRDD = spark.sparkContext.parallelize(["3 Rongcheng M 26","4 Guanhua M 27"]).map(lambda line : line.split(" "))
# rowRDD = studentRDD.map(lambda p : Row(p[1].strip(), p[2].strip(),int(p[3])))

# schema = StructType([StructField("name", StringType(), True),StructField("gender", StringType(), True),StructField("age",IntegerType(), True)])

# studentDF = spark.createDataFrame(rowRDD, schema)
# studentDF.registerTempTable("tempTable")
# hive_context.sql('insert into student select * from tempTable')




# Spark MLlib
# 构建一个机器学习工作流

# SparkSession
# SparkSession 功能
# 1：创建DataFrame
# 2：是 SparkSQL 的入口，然后可以基于 sparkSession 来获取或者是读取源数据来生成 DataFrame
spark = SparkSession.builder.master("local").appName("Word Count").getOrCreate()

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
 
training = spark.createDataFrame([
    (0, "a b c d e spark", 1.0),
    (1, "b d", 0.0),
    (2, "spark f g h", 1.0),
    (3, "hadoop mapreduce", 0.0)
], ["id", "text", "label"])


tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.001)

pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

model = pipeline.fit(training)

test = spark.createDataFrame([
    (4, "spark i j k"),
    (5, "l m n"),
    (6, "spark hadoop spark"),
    (7, "apache hadoop")
], ["id", "text"])

prediction = model.transform(test)
selected = prediction.select("id", "text", "probability", "prediction")
for row in selected.collect():
    rid, text, prob, prediction = row
    print("(%d, %s) --> prob=%s, prediction=%f" % (rid, text, str(prob), prediction))



# 1.特征抽取 — TF-IDF
from pyspark.ml.feature import HashingTF,IDF,Tokenizer
sentenceData = spark.createDataFrame([(0, "I heard about Spark and I love Spark")
                                    ,(0, "I wish Java could use case classes")
                                    ,(1, "Logistic regression models are neat")]).toDF("label", "sentence")


# 用tokenizer对句子进行分词
tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
wordsData = tokenizer.transform(sentenceData)

# 用HashingTF的transform()方法把句子哈希成特征向量
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(wordsData)

# 使用IDF来对单纯的词频特征向量进行修正
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)

# DFModel是一个Transformer，调用它的transform()方法，即可得到每一个单词对应的TF-IDF度量值
rescaledData = idfModel.transform(featurizedData)
rescaledData.select("label", "features").show()


# 3.特征抽取–CountVectorizer
# CountVectorizer旨在通过计数来将一个文档转换为向量。
from pyspark.ml.feature import CountVectorizer

df = spark.createDataFrame([(0, "I heard about Spark and I love Spark".split(" "))
                            ,(0, "I wish Java love heard case Spark".split(" "))
                            ,(1, "Spark Java models heard case".split(" "))], ["id", "words"])


# fit a CountVectorizerModel from the corpus.
cv = CountVectorizer(inputCol="words", outputCol="features", vocabSize=3, minDF=2.0)
model = cv.fit(df)

result = model.transform(df)
result.show()
result.show(truncate=False)



# 逻辑回归
# 1. 导入需要的包：
from pyspark.sql import Row,functions
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vector,Vectors
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer,HashingTF, Tokenizer
from pyspark.ml.classification import LogisticRegression,LogisticRegressionModel,BinaryLogisticRegressionSummary, LogisticRegression


# 2. 读取数据，简要分析：
def f(x):
    rel = {}
    rel['features'] = Vectors.dense(float(x[0]),float(x[1]),float(x[2]),float(x[3]))
    rel['label'] = str(x[4])
    return rel

# 加“*”时，函数可接受任意多个参数，全部放入一个元祖中
# 加“**”时，函数接受参数时，返回为字典，需要写为"k1=123"

data = spark.sparkContext.textFile("file:///home/dev/hejinyang/iris.txt").map(lambda line: line.split(',')).map(lambda p: Row(**f(p))).toDF()
data.show()

# 从中选出两类的数据
df=data.filter(data.label != 'Iris-setosa')


# 3. 构建ML的pipeline
labelIndexer = StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(df)
featureIndexer = VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").fit(df)

# labelIndexer1 = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(df)
# featureIndexer1 = VectorIndexer(inputCol="features", outputCol="indexedFeatures").fit(df)

# labelIndexer.transform(df).show()
# labelIndexer1.transform(df).show()
# featureIndexer.transform(df).show()
# featureIndexer1.transform(df).show()


# 冒号声明变量类型
# bbb:str = 'hddd'
# featureIndexer:org.apache.spark.ml.feature.VectorIndexerModel = vecIdx_53b988077b38

trainingData, testData = df.randomSplit([0.7,0.3])


lr = LogisticRegression().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
print("LogisticRegression parameters:\n" + lr.explainParams())

labelConverter = IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)


lrPipeline =  Pipeline().setStages([labelIndexer, featureIndexer, lr, labelConverter])
lrPipelineModel = lrPipeline.fit(trainingData)

lrPredictions = lrPipelineModel.transform(testData)

preRel = lrPredictions.select("predictedLabel", "label", "features", "probability").collect()
for item in preRel:
    print(str(item['label'])+','+str(item['features'])+'-->prob='+str(item['probability'])+',predictedLabel'+str(item['predictedLabel']))
 
Iris-versicolor,[5.2,2.7,3.9,1.4]-->prob=[0.474125433289,0.525874566711],predictedLabelIris-virginica
Iris-versicolor,[5.5,2.3,4.0,1.3]-->prob=[0.498724224708,0.501275775292],predictedLabelIris-virginica
Iris-versicolor,[5.6,3.0,4.5,1.5]-->prob=[0.456659495584,0.543340504416],predictedLabelIris-virginica


# 4.模型评估
evaluator = MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction")
lrAccuracy = evaluator.evaluate(lrPredictions)
print("Test Error = " + str(1.0 - lrAccuracy))


lrModel = lrPipelineModel.stages[2]
print("Coefficients: " + str(lrModel.coefficients)+"Intercept: "+str(lrModel.intercept)+"numClasses: "+str(lrModel.numClasses)+"numFeatures: "+str(lrModel.numFeatures))
Coefficients: [-0.0252899565937,0.0,0.0,0.0761866331438]Intercept: -0.0158571012943numClasses: 2numFeatures: 4


# 5.模型参数
trainingSummary = lrModel.summary
objectiveHistory = trainingSummary.objectiveHistory
for item in objectiveHistory:
...     print(item)
... 
0.6930582890371242
0.6899151958544979
0.6884360489604017
0.6866214680339037
0.6824264404293411
0.6734525297891238
0.6718869589782477
0.6700321119842002
0.6681741952485035
0.6668744860924799
0.6656055740433819
 
 
print(trainingSummary.areaUnderROC)
0.9889758179231863

# fMeasure是一个综合了召回率和准确率的指标，通过最大化fMeasure，我们可以选取到用来分类的最合适的阈值
fMeasure = trainningSummary.fMeasureByThreshold

maxFMeasure = fMeasure.select(functions.max("F-Measure")).head()[0]
0.9599999999999999
 
bestThreshold = fMeasure.where(fMeasure["F-Measure"]== maxFMeasure).select("threshold").head()[0]
0.5487261156903904
 
lr.setThreshold(bestThreshold)
