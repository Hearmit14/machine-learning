pyspark --master yarn --driver-memory 4g --driver-maxResultSize 4g

# 用hive数据创建dataframe
from pyspark import SparkConf
from pyspark.sql import HiveContext
from pyspark.sql import Row,functions
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vector,Vectors
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer,HashingTF, Tokenizer
from pyspark.ml.classification import LogisticRegression,LogisticRegressionModel,BinaryLogisticRegressionSummary, LogisticRegression
from sklearn.grid_search import GridSearchCV

spark=SparkSession.builder.config("spark.debug.maxToStringFields", "100").getOrCreate()
conf = SparkConf().set("spark.driver.memory", "4g").set("spark.driver.maxResultSize", "4g")
sc = spark.sparkContext(conf)


hive_context = HiveContext(sc)
hiveDataFrame=hive_context.sql('select * from sdk_user.hejy_temp_model_22002015')
predictDataFrame=hive_context.sql('select * from sdk_user.hejy_temp_model_all')


hiveDataFrame.select(hiveDataFrame.dhid,hiveDataFrame.app_1).show()
predictDataFrame.select(predictDataFrame.dhid,predictDataFrame.app_1).show()



# dataset也提供了转化RDD的操作。因此只需要将之前dataframe.map在中间修改为：dataframe.rdd.map即可。
def f(x):
    rel = {}
    rel['features'] = Vectors.dense(x[2:])
    rel['label'] = float(x[0])
    return rel

data = hiveDataFrame.rdd.map(lambda p: Row(**f(p))).toDF()
trainingData, testData = data.randomSplit([0.7,0.3])
data.show()


lr = LogisticRegression().setMaxIter(1000).setRegParam(0.1).setElasticNetParam(0.01)

lrModel = lr.fit(trainingData)

def g(x):
    rel = {}
    rel['features'] = Vectors.dense(x[1:3198])
    rel['id'] = str(x[0])
    return rel

predictdata = predictDataFrame.rdd.map(lambda p: Row(**g(p))).toDF()



result = lrModel.transform(predictdata)
result.first()

drop table if exists sdk_user.hejy_temp_3;
create table sdk_user.hejy_temp_3
(
dhid                     string,
rawPrediction                    string,
probability                     string,
prediction                float
)
row format delimited
fields terminated by '\t'
ESCAPED BY '\\' 
STORED AS textfile;



from pyspark.sql.types import Row
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql import HiveContext

result.registerTempTable("tempTable")

hive_context.sql('select * from tempTable limit 2').show()

hive_context.sql('insert into sdk_user.hejy_temp_3 select id,rawPrediction,probability,prediction from tempTable limit 100')


# lrPredictions = lr1.transform(testData)
# evaluator = MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction")
# lrAccuracy = evaluator.evaluate(lrPredictions)
# print("Test Error = " + str(1.0 - lrAccuracy))

lrModel.interceptVector
lrModel.coefficients
lrModel.coefficients.size
lrModel.coefficients.indices
lrModel.coefficients.values

trainingSummary = lr1.summary
print(trainingSummary.areaUnderROC)


# 查看网格搜索功能
# 不能用
gridlr = LogisticRegression().getParam('regParam')
parameters = dict(regParam = [0.01,0.1],elasticNetParam = [0.01,0.1])

grid = GridSearchCV(estimator=gridlr,param_grid=parameters,scoring="roc_auc",cv=5)


# 不能用dataframe来网络搜索，要数组，后续再看
trainingData1 = trainingData.collect()

gridlrModel = grid.fit(trainingData1)

# 把dataframe转为索引,没用
# labelIndexer = StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(trainingData)
# featureIndexer = VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").fit(trainingData)

# trainingData1 = labelIndexer.transform(trainingData)
# trainingData2 = featureIndexer.transform(trainingData1)

# trainingData2.show()

# gridlrModel = grid.fit(trainingData2)



# 使用spark的交叉验证来做
from pyspark.ml.linalg import Vector,Vectors
from pyspark.ml.feature import HashingTF,Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import Row
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.classification import LogisticRegression,LogisticRegressionModel
from pyspark.ml import Pipeline, PipelineModel

lr = LogisticRegression().setMaxIter(100)
paramGrid = ParamGridBuilder().addGrid(lr.elasticNetParam, [0.2,0.8]).addGrid(lr.regParam, [0.01, 0.1, 0.5]).build()
# 对于回归问题评估器可选择RegressionEvaluator，二值数据可选择BinaryClassificationEvaluator，多分类问题可选择MulticlassClassificationEvaluator。
cv = CrossValidator().setEstimator(lr).setEvaluator(BinaryClassificationEvaluator()).setEstimatorParamMaps(paramGrid).setNumFolds(3) 
cvModel = cv.fit(trainingData)

bestModel= cvModel.bestModel

bestModel.coefficientMatrix


