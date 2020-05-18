// 第6章 Spark MLlib
// 6.2 机器学习工作流
// 构建一个机器学习工作流
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().
            master("local").
            appName("my App Name").
            getOrCreate()

import spark.implicits._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.{Pipeline,PipelineModel}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
 
 
val training = spark.createDataFrame(Seq(
       (0L, "a b c d e spark", 1.0),
       (1L, "b d", 0.0),
       (2L, "spark f g h", 1.0),
       (3L, "hadoop mapreduce", 0.0)
     )).toDF("id", "text", "label")

// training: org.apache.spark.sql.DataFrame = [id: bigint, text: string, label: double]

val tokenizer = new Tokenizer().
      setInputCol("text").
      setOutputCol("words")

// val aa=tokenizer.transform(training)
 
val hashingTF = new HashingTF().
      setNumFeatures(20).
      setInputCol(tokenizer.getOutputCol).
      setOutputCol("features")

// val bb=hashingTF.transform(aa)
// bb.select(bb("features")).show(false)
 
val lr = new LogisticRegression().
      setMaxIter(10).
      setRegParam(0.01)

val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, lr))

val model = pipeline.fit(training)

val test = spark.createDataFrame(Seq(
     (4L, "spark i j k"),
     (5L, "l m n"),
     (6L, "spark a"),
     (7L, "apache hadoop")
   )).toDF("id", "text")

model.transform(test).
      select("id", "text", "probability", "prediction").
      collect().
      foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
        println(s"($id, $text) --> prob=$prob, prediction=$prediction")
      }

(4, spark i j k) --> prob=[0.5406433544851421,0.45935664551485783], prediction=0.0
(5, l m n) --> prob=[0.9334382627383259,0.06656173726167405], prediction=0.0
(6, spark a) --> prob=[0.15041430048068286,0.8495856995193171], prediction=1.0
(7, apache hadoop) --> prob=[0.9768636139518304,0.023136386048169585], prediction=0.0


// 6.3 特征抽取、转化和选择
// 6.3.1 特征抽取：TF-IDF
// TF-IDF (HashingTF and IDF)
// ​在Spark ML库中，TF-IDF被分成两部分：TF (+hashing) 和 IDF。
// TF: HashingTF 是一个Transformer，在文本处理中，接收词条的集合然后把这些集合转化成固定长度的特征向量。这个算法在哈希的同时会统计各个词条的词频。

// IDF: IDF是一个Estimator，在一个数据集上应用它的fit（）方法，产生一个IDFModel。 
// 该IDFModel 接收特征向量（由HashingTF产生），然后计算每一个词在文档中出现的频次。IDF会减少那些在语料库中出现频率较高的词的权重。
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
val sentenceData = spark.createDataFrame(Seq(
      (0, "I heard about Spark and I love Spark"),
      (0, "I wish Java could use case classes"),
      (1, "Logistic regression models are neat")
    )).toDF("label", "sentence")
// eData: org.apache.spark.sql.DataFrame = [label: int, sentence: string]

val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
 
val wordsData = tokenizer.transform(sentenceData)

wordsData.show(false)

val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(2000)
 
val featurizedData = hashingTF.transform(wordsData)

featurizedData.select("rawFeatures").show(false)
// ​可以看到，分词序列被变换成一个稀疏特征向量，其中每个单词都被散列成了一个不同的索引值，特征向量在某一维度上的值即该词汇在文档中出现的次数。

val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")

val idfModel = idf.fit(featurizedData)

// 很显然，IDFModel是一个Transformer，调用它的transform()方法，即可得到每一个单词对应的TF-IDF度量值
val rescaledData = idfModel.transform(featurizedData)

rescaledData.select("features", "label").take(3).foreach(println)

// 6.3.2 特征抽取：Word2Vec
import org.apache.spark.ml.feature.Word2Vec
 
val documentDF = spark.createDataFrame(Seq(
  "Hi I heard about Spark".split(" "),
  "I wish Java could use case classes".split(" "),
  "Logistic regression models are neat".split(" ")
).map(Tuple1.apply)).toDF("text")


val word2Vec = new Word2Vec().
  setInputCol("text").
  setOutputCol("result").
  setVectorSize(3).
  setMinCount(0)

val model = word2Vec.fit(documentDF)

val result = model.transform(documentDF)

result.select("result").take(3).foreach(println)
// [[0.018490654602646827,-0.016248732805252075,0.04528368394821883]]
// [[0.05958533100783825,0.023424440695505054,-0.027310076036623544]]
// [[-0.011055880039930344,0.020988055132329465,0.042608972638845444]]

// 6.3.3 特征抽取：CountVectorizer
// CountVectorizer旨在通过计数来将一个文档转换为向量。
// 当不存在先验字典时，Countvectorizer作为Estimator提取词汇进行训练，并生成一个CountVectorizerModel用于存储相应的词汇向量空间。
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}

val df = spark.createDataFrame(Seq(
    (0, Array("a", "b", "c")),
    (1, Array("a", "b", "b", "c", "a"))
  )).toDF("id", "words")

val cvModel: CountVectorizerModel = new CountVectorizer().
    setInputCol("words").
    setOutputCol("features").
    setVocabSize(30).
    setMinDF(2).
    fit(df)

// 在训练结束后，可以通过CountVectorizerModel的vocabulary成员获得到模型的词汇表：
cvModel.vocabulary
res7: Array[String] = Array(b, a, c)

cvModel.transform(df).show(false)
+---+---------------+-------------------------+
|id |words          |features                 |
+---+---------------+-------------------------+
|0  |[a, b, c]      |(3,[0,1,2],[1.0,1.0,1.0])|
|1  |[a, b, b, c, a]|(3,[0,1,2],[2.0,2.0,1.0])|
+---+---------------+-------------------------+

// 6.3.4 特征变换：标签和索引的转化
// 在机器学习处理过程中，为了方便相关算法的实现，经常需要把标签数据（一般是字符串）转化成整数索引，或是在计算结束后将整数索引还原为相应的标签。
// Spark ML包中提供了几个相关的转换器，例如：StringIndexer、IndexToString、OneHotEncoder、VectorIndexer，它们提供了十分方便的特征转换功能
// 1 StringIndexer
// ​StringIndexer转换器可以把一列类别型的特征（或标签）进行编码，使其数值化，索引的范围从0开始，该过程可以使得相应的特征索引化
// 索引构建的顺序为标签的频率，优先编码频率较大的标签，所以出现频率最高的标签为0号。
// 如果输入的是数值型的，我们会把它转化成字符型，然后再对其进行编码。

import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}
 
val df1 = spark.createDataFrame(Seq(
          (0, "a"),
          (1, "b"),
          (2, "c"),
          (3, "a"),
          (4, "a"),
          (5, "c"))).toDF("id", "category")

val indexer = new StringIndexer().
  setInputCol("category").
  setOutputCol("categoryIndex")
 
val model = indexer.fit(df1)

val indexed1 = model.transform(df1)
 
indexed1.show()
+---+--------+-------------+
| id|category|categoryIndex|
+---+--------+-------------+
|  0|       a|          0.0|
|  1|       b|          2.0|
|  2|       c|          1.0|
|  3|       a|          0.0|
|  4|       a|          0.0|
|  5|       c|          1.0|
+---+--------+-------------+

// 考虑这样一种情况，我们使用已有的数据构建了一个StringIndexerModel，然后再构建一个新的DataFrame，这个DataFrame中有着模型内未曾出现的标签“d”，用已有的模型去转换这一DataFrame会有什么效果？
// 实际上，如果直接转换的话，Spark会抛出异常，报出“Unseen label: d”的错误。
// 为了处理这种情况，在模型训练后，可以通过设置setHandleInvalid("skip")来忽略掉那些未出现的标签，这样，带有未出现标签的行将直接被过滤掉
val df2 = spark.createDataFrame(Seq(
                (0, "a"),
                (1, "b"),
                (2, "c"),
                (3, "a"),
                (4, "a"),
                (5, "d"))).toDF("id", "category")

val indexed = model.transform(df2)

indexed.show()
 
val indexed2 = model.setHandleInvalid("skip").transform(df2)
 
indexed2.show()
+---+--------+-------------+
| id|category|categoryIndex|
+---+--------+-------------+
|  0|       a|          0.0|
|  1|       b|          2.0|
|  2|       c|          1.0|
|  3|       a|          0.0|
|  4|       a|          0.0|
+---+--------+-------------+
 
// 2 IndexToString
// ​与StringIndexer相对应，IndexToString的作用是把标签索引的一列重新映射回原有的字符型标签。
val converter = new IndexToString().
  setInputCol("categoryIndex").
  setOutputCol("originalCategory")
 
val converted = converter.transform(indexed)
 
scala> converted.select("id", "originalCategory").show()
+---+----------------+
| id|originalCategory|
+---+----------------+
|  0|               a|
|  1|               b|
|  2|               c|
|  3|               a|
|  4|               a|
|  5|               c|
+---+----------------+

// 3 OneHotEncoder
// ​独热编码（One-Hot Encoding） 是指把一列类别性特征（或称名词性特征，nominal/categorical features）映射成一系列的二元连续特征的过程，
// 原有的类别性特征有几种可能取值，这一特征就会被映射成几个二元连续特征，每一个特征代表一种取值，若该样本表现出该特征，则取1，否则取0。
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
 
val df = spark.createDataFrame(Seq(
  (0, "a"),
  (1, "b"),
  (2, "c"),
  (3, "a"),
  (4, "a"),
  (5, "c"),
  (6, "d"),
  (7, "d"),
  (8, "d"),
  (9, "d"),
  (10, "e"),
  (11, "e"),
  (12, "e"),
  (13, "e"),
  (14, "e")
)).toDF("id", "category")
 
val indexer = new StringIndexer().
  setInputCol("category").
  setOutputCol("categoryIndex").
  fit(df)
 
val indexed = indexer.transform(df)

val encoder = new OneHotEncoder().
  setInputCol("categoryIndex").
  setOutputCol("categoryVec").
  setDropLast(false)
 
val encoded = encoder.transform(indexed)
 
encoded.show()

// 4 VectorIndexer
// 之前介绍的StringIndexer是针对单个类别型特征进行转换，倘若所有特征都已经被组织在一个向量中，又想对其中某些单个分量进行处理时，Spark ML提供了VectorIndexer类来解决向量数据集中的类别性特征转换。
// 通过为其提供maxCategories超参数，它可以自动识别哪些特征是类别型的，并且将原始值转换为类别索引。它基于不同特征值的数量来识别哪些特征需要被类别化，那些取值可能性最多不超过maxCategories的特征需要会被认为是类别型的。
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.linalg.{Vector, Vectors}
 
val data = Seq(
    Vectors.dense(-1.0, 1.0, 1.0),
    Vectors.dense(-1.0, 3.0, 1.0),
    Vectors.dense(0.0, 5.0, 1.0),
    Vectors.dense(0.0, 2.0, 3.0),
    Vectors.dense(1.0, 4.0, 5.0)
    )

val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
 
val indexer = new VectorIndexer().
  setInputCol("features").
  setOutputCol("indexed").
  setMaxCategories(2)
 
val indexerModel = indexer.fit(df)

val categoricalFeatures: Set[Int] = indexerModel.categoryMaps.keys.toSet
 
println(s"Chose ${categoricalFeatures.size} categorical features: " + categoricalFeatures.mkString(", "))

val indexed = indexerModel.transform(df)
 
indexed.show()
+--------------+-------------+
|      features|      indexed|
+--------------+-------------+
|[-1.0,1.0,1.0]|[1.0,1.0,0.0]|
|[-1.0,3.0,1.0]|[1.0,3.0,0.0]|
| [0.0,5.0,1.0]|[0.0,5.0,0.0]|
+--------------+-------------+

// 6.3.5 特征选取：卡方选择器
import org.apache.spark.ml.feature.{ChiSqSelector, ChiSqSelectorModel}
import org.apache.spark.ml.linalg.Vectors

val df = spark.createDataFrame(Seq(
(1, Vectors.dense(0.0, 0.0, 18.0, 1.0), 1),
(2, Vectors.dense(0.0, 1.0, 12.0, 0.0), 0),
(3, Vectors.dense(1.0, 0.0, 15.0, 0.1), 0)
)).toDF("id", "features", "label")
// df.printSchema()
val df1 = spark.createDataFrame(Array(
(1, Vectors.dense(0.0, 0.0, 18.0, 1.0), 1),
(2, Vectors.dense(0.0, 1.0, 12.0, 0.0), 0),
(3, Vectors.dense(1.0, 0.0, 15.0, 0.1), 0)
)).toDF("id", "features", "label")
df1.printSchema()

df.show()
+---+------------------+-----+
| id|          features|label|
+---+------------------+-----+
|  1|[0.0,0.0,18.0,1.0]|    1|
|  2|[0.0,1.0,12.0,0.0]|    0|
|  3|[1.0,0.0,15.0,0.1]|    0|
+---+------------------+-----+

val selector = new ChiSqSelector().
setNumTopFeatures(1).
setFeaturesCol("features").
setLabelCol("label").
setOutputCol("selected-feature")
 
val selector_model = selector.fit(df)

val selector_model = selector.fit(df)
 
val result = selector_model.transform(df)
 
result.show(false)
+---+------------------+-----+----------------+
|id |features          |label|selected-feature|
+---+------------------+-----+----------------+
|1  |[0.0,0.0,18.0,1.0]|1.0  |[18.0]          |
|2  |[0.0,1.0,12.0,0.0]|0.0  |[12.0]          |
|3  |[1.0,0.0,15.0,0.1]|0.0  |[15.0]          |
+---+------------------+-----+----------------+


// 第12.2.1节 MLlib基本数据类型
// 一、本地向量（Local Vector）
// 稠密向量使用一个双精度浮点型数组来表示其中每一维元素，而稀疏向量则是基于一个整型索引数组和一个双精度浮点型的值数组。
// 例如，向量(1.0, 0.0, 3.0)的稠密向量表示形式是[1.0,0.0,3.0]，而稀疏向量形式则是(3, [0,2], [1.0, 3.0])，其中，3是向量的长度，[0,2]是向量中非0维度的索引值，表示位置为0、2的两个元素为非零值，而[1.0, 3.0]则是按索引排列的数组元素值。
import org.apache.spark.mllib.linalg.{Vector, Vectors}
 
// 创建一个稠密本地向量
val dv: Vector = Vectors.dense(2.0, 0.0, 8.0)
// 创建一个稀疏本地向量
// 方法第二个参数数组指定了非零元素的索引，而第三个参数数组则给定了非零元素值
val sv1: Vector = Vectors.sparse(3, Array(0, 2), Array(2.0, 8.0))
// 另一种创建稀疏本地向量的方法
// 方法的第二个参数是一个序列，其中每个元素都是一个非零值的元组：(index,elem)
val sv2: Vector = Vectors.sparse(3, Seq((0, 2.0), (2, 8.0)))

// 二、标注点（Labeled Point）
// 标注点LabeledPoint是一种带有标签（Label/Response）的本地向量，它可以是稠密或者是稀疏的。在MLlib中，标注点在监督学习算法中被使用。
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
//创建一个标签为1.0（分类中可视为正样本）的稠密向量标注点
val pos = LabeledPoint(1.0, org.apache.spark.mllib.linalg.Vectors.dense(2.0, 0.0, 8.0))
//创建一个标签为0.0（分类中可视为负样本）的稀疏向量标注点
val neg = LabeledPoint(0.0, org.apache.spark.mllib.linalg.Vectors.sparse(3, Array(0, 2), Array(2.0, 8.0)))

// 在实际的机器学习问题中，稀疏向量数据是非常常见的，MLlib提供了读取LIBSVM格式数据的支持，该格式被广泛用于LIBSVM、LIBLINEAR等机器学习库。在该格式下，每一个带标注的样本点由以下格式表示：
import org.apache.spark.mllib.util.MLUtils
 
// 用loadLibSVMFile方法读入LIBSVM格式数据
// sample_libsvm_data.txt为spark自带的一个示例，在以下地址可以找到：
// $SPARK_HOME$/data/mllib/sample_libsvm_data.txt
val examples = MLUtils.loadLibSVMFile(sc, "/data/mllib/sample_libsvm_data.txt")
//返回的是组织成RDD的一系列LabeledPoint

// 三、本地矩阵（Local Matrix）
// MLlib支持稠密矩阵DenseMatrix和稀疏矩阵Sparse Matrix两种本地矩阵
// 稠密矩阵将所有元素的值存储在一个列优先（Column-major）的双精度型数组中，而稀疏矩阵则将非零元素以列优先的CSC（Compressed Sparse Column）模式进行存储 
import org.apache.spark.mllib.linalg.{Matrix, Matrices}

// 创建一个3行2列的稠密矩阵[ [1.0,2.0], [3.0,4.0], [5.0,6.0] ]
// 请注意，这里的数组参数是列先序的！
val dm: Matrix = Matrices.dense(3, 2, Array(1.0, 3.0, 5.0, 2.0, 4.0, 6.0))
1.0  2.0
3.0  4.0
5.0  6.0

// 创建一个3行2列的稀疏矩阵[ [9.0,0.0], [0.0,8.0], [0.0,6.0]]
// 第一个数组参数表示列指针，即每一列元素的开始索引值
// 第二个数组参数表示行索引，即对应的元素是属于哪一行
// 第三个数组即是按列先序排列的所有非零元素，通过列指针和行索引即可判断每个元素所在的位置
val sm: Matrix = Matrices.sparse(3, 2, Array(0, 1, 3), Array(0, 2, 1), Array(9, 6, 8))
(0,0) 9.0
(2,1) 6.0
(1,1) 8.0
 
// 四、分布式矩阵（Distributed Matrix）
// 行矩阵RowMatrix，索引行矩阵IndexedRowMatrix、坐标矩阵CoordinateMatrix和分块矩阵Block Matrix。
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector,Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
 
// 创建两个本地向量dv1 dv2
val dv1 : Vector = Vectors.dense(1.0,2.0,3.0)
val dv2 : Vector = Vectors.dense(2.0,3.0,4.0)
// 使用两个本地向量创建一个RDD[Vector]
val rows : RDD[Vector] = sc.parallelize(Array(dv1,dv2))
 
// 通过RDD[Vector]创建一个行矩阵
val mat : RowMatrix = new RowMatrix(rows)
//可以使用numRows()和numCols()方法得到行数和列数
mat.numRows()
res0: Long = 2
mat.numCols()
res1: Long = 3
mat.rows.foreach(println)
[1.0,2.0,3.0]
[2.0,3.0,4.0]


// 第12.2.3节 基本的统计工具
// 二、摘要统计 Summary statistics
// val aa=sc.textFile("file:///home/hejy/iris.data")
// val aa=spark.read.option("delimiter","\t").option("header","true").csv("home/hejy/iris.data")
import org.apache.spark.mllib.linalg.{Vector,Matrix}
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}

val observations=sc.textFile("/user/hejy/iris.data").filter(line=>line!="").map(_.split(",")).map(p => org.apache.spark.mllib.linalg.Vectors.dense(p(0).toDouble, p(1).toDouble, p(2).toDouble, p(3).toDouble))

val summary: MultivariateStatisticalSummary = Statistics.colStats(observations)
println(summary.count)
150
println(summary.mean)
[5.843333333333332,3.0540000000000003,3.7586666666666666,1.1986666666666668]
println(summary.variance)
[0.685693512304251,0.18800402684563744,3.113179418344516,0.5824143176733783]
println(summary.max)
[7.9,4.4,6.9,2.5]
println(summary.min)
[4.3,2.0,1.0,0.1]
println(summary.normL1)
[876.4999999999998,458.1000000000001,563.8000000000002,179.79999999999995]
println(summary.normL2)
[72.27620631992245,37.77631533117014,50.82322303829225,17.38677658451963]
println(summary.numNonzeros)
[150.0,150.0,150.0,150.0]

// 三、相关性Correlations
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.stat.Statistics

val seriesX = observations.map(p => p(0).toDouble) 
val seriesY = observations.map(p => p(1).toDouble) 

val correlation: Double = Statistics.corr(seriesX, seriesY, "pearson")

val correlMatrix1: Matrix = Statistics.corr(observations, "pearson")

// 四、分层抽样 Stratified sampling
// （一）sampleByKey 方法
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.PairRDDFunctions

val data = sc.makeRDD(Array(
    ("female","Lily"),
    ("female","Lucy"),
    ("female","Emily"),
    ("female","Kate"),
    ("female","Alice"),
    ("male","Tom"),
    ("male","Roy"),
    ("male","David"),
    ("male","Frank"),
    ("male","Jack")))

val fractions : Map[String, Double]= Map("female"->0.6,"male"->0.4)

val approxSample = data.sampleByKey(withReplacement = false, fractions, 1)
approxSample.collect()


// （二）sampleByKeyExact 方法
val exactSample = data.sampleByKeyExact(withReplacement = false, fractions, 1)
exactSample.collect()

// 五、假设检验 Hypothesis testing
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.Statistics._

// 拟合度检验要求输入为Vector, 独立性检验要求输入是Matrix。
// (一) 适合度检验 Goodness fo fit
// Goodness fo fit（适合度检验）：验证一组观察值的次数分配是否异于理论上的分配。
val v1: Vector = observations.first
val v2: Vector = observations.take(2).last

val goodnessOfFitTestResult = Statistics.chiSqTest(v1)

Chi squared test summary:
method: pearson
degrees of freedom = 3
statistic = 5.588235294117647
pValue = 0.1334553914430291
No presumption against null hypothesis: observed follows the same distribution as expected

// （二）独立性检验 Indenpendence
// 我们通过v1、v2构造一个举证Matrix，然后进行独立性检验
val mat: Matrix =Matrices.dense(2,2,Array(v1(0),v1(1),v2(0),v2(1)))
val a =Statistics.chiSqTest(mat)

// 把v1作为样本，把v2作为期望值，进行卡方检验
val c1 = Statistics.chiSqTest(v1, v2)

// 键值对也可以进行独立性检验
val obs=sc.textFile("/user/hejy/iris.data").filter(_!="").map(_.split(",")).map{parts =>
    LabeledPoint(if(parts(4)=="Iris-setosa") 0.toDouble else if (parts(4)=="Iris-versicolor") 1.toDouble else 2.toDouble,
                Vectors.dense(parts(0).toDouble,parts(1).toDouble,parts(2).toDouble,parts(3).toDouble)
    )
}

val featureTestResults= Statistics.chiSqTest(obs)

// 六、随机数生成 Random data generation
import org.apache.spark.SparkContext
import org.apache.spark.mllib.random.RandomRDDs._

val u = normalRDD(sc, 10000000L, 10)
val v = u.map(x => 1.0 + 2.0 * x)

// 七、核密度估计 Kernel density estimation
// ​ Spark ML 提供了一个工具类 KernelDensity 用于核密度估算，核密度估算的意思是根据已知的样本估计未知的密度，属於非参数检验方法之一
import org.apache.spark.mllib.stat.KernelDensity
import org.apache.spark.rdd.RDD

val test = sc.textFile("G:/spark/iris.data").map(_.split(",")).map(p => p(0).toDouble)
val densities = kd.estimate(Array(-1.0, 2.0, 5.0, 5.8))

// 第12.2.5节 降维操作
// 第12.2.5.1节 奇异值分解（SVD）
import org.apache.spark.mllib.linalg.distributed.RowMatrix

// val mat =  sc.parallelize(Array(Array(1,2,3,4,5,6,7,8,9),
//                 Array(5,6,7,8,9,0,8,6,7),
//                 Array(9,0,8,7,1,4,3,2,1),
//                 Array(6,4,2,1,3,4,2,1,5)
//                 )
// ).map(line=>Vectors.dense(line(0).toDouble,line(1).toDouble,line(2).toDouble,line(3).toDouble,line(4).toDouble,line(5).toDouble,line(6).toDouble,line(7).toDouble,line(8).toDouble))

val mat =  sc.parallelize(Array(Array(1,2,3,4,5,6,7,8,9),
                Array(5,6,7,8,9,0,8,6,7),
                Array(9,0,8,7,1,4,3,2,1),
                Array(6,4,2,1,3,4,2,1,5)
                )
).map(line=>line.map(_.toDouble))

val data=mat.map(line => Vectors.dense(line))
//通过RDD[Vectors]创建行矩阵
val rm = new RowMatrix(data)
//保留前3个奇异值
val svd = rm.computeSVD(3)
svd.s
svd.V
svd.U

// 第12.2.5.2节 主成分分析（PCA）
// MLlib提供了两种进行PCA变换的方法，第一种与上文提到的SVD分解类似，位于org.apache.spark.mllib.linalg包下的RowMatrix中
scala> import org.apache.spark.mllib.linalg.Vectors
scala> import org.apache.spark.mllib.linalg.distributed.RowMatrix

val rm = new RowMatrix(data)
//保留前3个主成分
val pc = rm.computePrincipalComponents(3)

val projected = rm.multiply(pc)

// MLlib还提供了一种“模型式”的PCA变换实现，它位于org.apache.spark.mllib.feature包下的PCA类，它可以接受RDD[Vectors]作为参数，进行PCA变换。
// 该方法特别适用于原始数据是LabeledPoint类型的情况，只需取出LabeledPoint的feature成员（它是RDD[Vector]类型），对其做PCA操作后再放回，即可在不影响原有标签情况下进行PCA变换。
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.mllib.regression.LabeledPoint
val data = mat.map(line => {
    LabeledPoint( if(line(0) > 1.0) 1.toDouble else 0.toDouble, Vectors.dense(line) )
})

val pca = new PCA(3).fit(data.map(_.features))

val projected = data.map(p => p.copy(features = pca.transform(p.features)))
 
projected.foreach(println)
