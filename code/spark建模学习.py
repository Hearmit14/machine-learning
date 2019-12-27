# 用hive数据创建dataframe
from pyspark.sql import Row,functions,HiveContext
from pyspark.ml.linalg import Vector,Vectors
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer,HashingTF, Tokenizer
from pyspark.ml.classification import LogisticRegression,LogisticRegressionModel,BinaryLogisticRegressionSummary, LogisticRegression


hive_context = HiveContext(sc)
hiveDataFrame=hive_context.sql('select * from sdk_user.hejy_temp_model_22002015')

hiveDataFrame.select(hiveDataFrame.dhid,hiveDataFrame.app_1).show()


def f(x):
    rel = {}
    rel['features'] = Vectors.dense(x[2:])
    rel['label'] = str(x[0])
    return rel

# dataset也提供了转化RDD的操作。因此只需要将之前dataframe.map在中间修改为：dataframe.rdd.map即可。
data = hiveDataFrame.rdd.map(lambda p: Row(**f(p))).toDF()
data.show()

trainingData, testData = data.randomSplit([0.7,0.3])

##############################################################################################################################
labelIndexer = StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
featureIndexer = VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").fit(data)

lr = LogisticRegression().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(1000).setRegParam(0.3).setElasticNetParam(0.0)

labelConverter = IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

# lrModel = lr.fit(trainingData)


lrPipeline =  Pipeline().setStages([labelIndexer, featureIndexer, lr, labelConverter])
lrPipelineModel = lrPipeline.fit(trainingData)

lrPredictions = lrPipelineModel.transform(testData)

evaluator = MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction")
lrAccuracy = evaluator.evaluate(lrPredictions)
print("Test Error = " + str(1.0 - lrAccuracy))


lrModel = lrPipelineModel.stages[2]
trainingSummary = lrModel.summary
objectiveHistory = trainingSummary.objectiveHistory

print(trainingSummary.areaUnderROC)


##############################################################################################################################

