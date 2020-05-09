// 第三章 Scala基础
// 声明变量
// Scala有两种类型的变量，一种是val，是不可变的，在声明时就必须被初始化，而且初始化以后就不能再赋值；另一种是var，是可变的，声明的时候需要进行初始化，初始化以后还可以再次对其赋值。
val myStr = "Hello World!"
val myStr2 : String = "Hello World!"
var myPrice : Double = 9.9

// Scala的数据类型包括：Byte、Char、Short、Int、Long、Float、Double和Boolean。
// 和Java不同的是，在Scala中，这些类型都是“类”，并且都是包scala的成员，比如，Int的全名是scala.Int。对于字符串，Scala用java.lang.String类来表示字符串。

// Scala允许对“字面量”直接执行方法
5.toString() //产生字符串"5"
"abc".intersect("bcd")  //输出"bc"

// 运算
// 在Scala中，可以使用加(+)、减(-) 、乘(*) 、除(/) 、余数（%）等操作符，而且，这些操作符就是方法。例如，5 + 3和(5).+(3)是等价的
scala> val sum1 = 5 + 3 //实际上调用了 (5).+(3)
sum1: Int = 8
scala> val sum2 = (5).+(3) //可以发现，写成方法调用的形式，和上面得到相同的结果
sum2: Int = 8

scala> var i = 5;
i: Int = 5
scala> i += 1  //将i递增
scala> println(i)
6

// range
// 在执行for循环时，我们经常会用到数值序列，比如，i的值从1循环到5，这时就可以采用Range来实现。
scala> 1 to 5
res0: scala.collection.immutable.Range.Inclusive = Range(1, 2, 3, 4, 5)
scala> 1.to(5)
res0: scala.collection.immutable.Range.Inclusive = Range(1, 2, 3, 4, 5)
scala> 1 until 5
res1: scala.collection.immutable.Range = Range(1, 2, 3, 4)
scala> 1 to 10 by 2
res2: scala.collection.immutable.Range = Range(1, 3, 5, 7, 9)
scala> 0.5f to 5.9f by 0.8f
res3: scala.collection.immutable.NumericRange[Float] = NumericRange(0.5, 1.3, 2.1, 2.8999999, 3.6999998, 4.5, 5.3)

// 打印语句
print("My name is:")
print("Ziyu")

println("My name is:")
println("Ziyu")


// 第四章 控制结构
// if
val x = 3
if (x>0) {
    println("This is a positive number")
} else if (x==0) {
    println("This is a zero")
} else {
    println("This is a negative number")
}

// Scala中的if表达式的值可以赋值给变量
val x = 6
val a = if (x>0) 1 else -1

// while
var i = 9
while (i > 0) {
    i -= 1
    printf("i is %d\n",i)
}

// for
for (i <- 1 to 5) println(i)
for (i <- 1 to 5 by 2) println(i)
for (i <- 1 to 5 if i%2==0) println(i)
for (i <- 1 to 5; j <- 1 to 3) println(i*j)


// 第五章 数据结构
// Scala:数组
val intValueArr = new Array[Int](3)  //声明一个长度为3的整型数组，每个数组元素初始化为0
intValueArr(0) = 12 //给第1个数组元素赋值为12
intValueArr(1) = 45  //给第2个数组元素赋值为45
intValueArr(2) = 33 //给第3个数组元素赋值为33

val intValueArr = Array(12,45,33)
val myStrArr = Array("BigData","Hadoop","Spark")

// 列表
val intList = List(1,2,3)

// 可以使用::操作，在列表的头部增加新的元素，得到一个新的列表
val intListOther = 0::intList

// ::操作符是右结合的，因此，如果要构建一个列表List(1,2,3)，实际上也可以采用下面的方式：
val intList = 1::2::3::Nil

val intList1 = List(1,2)
val intList2 = List(3,4)
val intList3 = intList1:::intList2


// 元组
// 元组是不同类型的值的聚集。元组和列表不同，列表中各个元素必须是相同类型，而元组可以包含不同类型的元素
scala> val tuple = ("BigData",2015,45.0)
tuple: (String, Int, Double) = (BigData,2015,45.0)  //这行是Scala解释器返回的执行结果
scala> println(tuple._1)
BigData
scala> println(tuple._2)
2015
scala> println(tuple._3)
45.0

// 集
// 集(set)是不重复元素的集合。列表中的元素是按照插入的先后顺序来组织的，但是，”集”中的元素并不会记录元素的插入顺序，而是以“哈希”方法对元素的值进行组织，所以，它允许你快速地找到某个元素
scala> var mySet = Set("Hadoop","Spark")
mySet: scala.collection.immutable.Set[String] = Set(Hadoop, Spark)
scala> mySet += "Scala"  //向mySet中增加新的元素
scala> println(mySet.contains("Scala"))
true

// 映射
// 映射(Map)是一系列键值对的集合，也就是，建立了键和值之间的对应关系。在映射中，所有的值，都可以通过键来获取。
val university = Map("XMU" -> "Xiamen University", "THU" -> "Tsinghua University","PKU"->"Peking University")
println(university("XMU"))
for ((k,v) <- university) printf("Code is : %s and name is: %s\n",k,v)

for (k<-university.keys) println(k)
XMU //打印出的结果
THU //打印出的结果
PKU //打印出的结果

// 迭代器
// 迭代器（Iterator）不是一个集合，但是，提供了访问集合的一种方法。当构建一个集合需要很大的开销时（比如把一个文件的所有行都读取内存），迭代器就可以发挥很好的作用。
// 迭代器包含两个基本操作：next和hasNext。next可以返回迭代器的下一个元素，hasNext用于检测是否还有下一个元素。

val iter = Iterator("Hadoop","Spark","Scala")
while (iter.hasNext) {
    println(iter.next())
}

for (elem <- iter) {
    println(elem)
}

// 第十一章 函数式编程
// 函数定义和高阶函数
def counter(value: Int): Int = { value += 1}

// 上面定义个这个函数的“类型”如下：
// (Int) => Int
// 实际上，我们只要把函数定义中的类型声明部分去除，剩下的就是函数的“值”，如下：
(value) => {value += 1} //只有一条语句时，大括号可以省略

// 现在，我们再按照大家比较熟悉的定义变量的方式，采用Scala语法来定义一个函数，如下：
// 声明一个变量时，我们采用的形式是：
val num: Int = 5 //当然，Int类型声明也可以省略，因为Scala具有自动推断类型的功能
// 照葫芦画瓢，我们也可以按照上面类似的形式来定义Scala中的函数：
val counter: Int => Int = { (value) => value += 1 }

scala> val myNumFunc: Int=>Int = (num: Int) => num * 2 //这行是我们输入的命令，把匿名函数定义为一个值，赋值给myNumFunc变量
myNumFunc: Int => Int = <function1>  //这行是执行返回的结果
scala> println(myNumFunc(3)) //myNumFunc函数调用的时候，需要给出参数的值，这里传入3，得到乘法结果是6
6

scala> val myNumFunc = (num: Int) => num * 2
myNumFunc: Int => Int = <function1>
scala> println(myNumFunc(3))
6

// 遍历操作
// 列表的遍历
val list = List(1, 2, 3, 4, 5)
for (elem <- list) println(elem)

val list = List(1, 2, 3, 4, 5)
list.foreach(elem => println(elem)) //本行语句甚至可以简写为list.foreach(println)，或者写成：list foreach println

// 映射的遍历
val university = Map("XMU" -> "Xiamen University", "THU" -> "Tsinghua University","PKU"->"Peking University")
for ((k,v) <- university) printf("Code is : %s and name is: %s\n",k,v)
for (k<-university.keys) println(k)
XMU //打印出的结果
THU //打印出的结果
PKU //打印出的结果

for (v<-university.values) println(v)
Xiamen University //打印出的结果
Tsinghua University //打印出的结果
Peking University  //打印出的结果


// map操作
// map操作是针对集合的典型变换操作，它将某个函数应用到集合中的每个元素，并产生一个结果集合
scala> val books = List("Hadoop", "Hive", "HDFS")
books: List[String] = List(Hadoop, Hive, HDFS)
scala> books.map(s => s.toUpperCase)
res0: List[String] = List(HADOOP, HIVE, HDFS)

// flatMap操作
// flatMap是map的一种扩展。在flatMap中，我们会传入一个函数，该函数对每个输入都会返回一个集合（而不是一个元素），然后，flatMap把生成的多个集合“拍扁”成为一个集合。
scala> val books = List("Hadoop","Hive","HDFS")
books: List[String] = List(Hadoop, Hive, HDFS)
scala> books flatMap (s => s.toList)
scala> books.flatMap(s => s.toList)
res0: List[Char] = List(H, a, o, o, p, H, i, v, e, H, D, F, S)


// 在实际编程中，我们经常会用到一种操作，遍历一个集合并从中获取满足指定条件的元素组成一个新的集合。Scala中可以通过filter操作来实现。
val university = Map("XMU" -> "Xiamen University", "THU" -> "Tsinghua University","PKU"->"Peking University","XMUT"->"Xiamen University of Technology")
val universityOfXiamen = university filter {kv => kv._2 contains "Xiamen"}

universityOfXiamen foreach {kv => println(kv._1+":"+kv._2)}

// reduce操作
// 在Scala中，我们可以使用reduce这种二元操作对集合中的元素进行归约。
scala> val list = List(1,2,3,4,5)
list: List[Int] = List(1, 2, 3, 4, 5)
scala> list.reduceLeft(_ + _)
res21: Int = 15
scala> list.reduceRight(_ + _)
res22: Int = 15

scala> val list = List(1,2,3,4,5)
list: List[Int] = List(1, 2, 3, 4, 5)
scala> list.reduceLeft(_ - _)
res25: Int = -13
scala> list.reduceRight(_ - _)
res26: Int = 3

// fold操作
// 折叠(fold)操作和reduce（归约）操作比较类似。fold操作需要从一个初始的“种子”值开始，并以该值作为上下文，处理集合中的每个元素。
scala> val list = List(1,2,3,4,5)
list: List[Int] = List(1, 2, 3, 4, 5)
 
scala> list.fold(10)(_*_)
res0: Int = 1200


// 第3章 Spark编程基础
// rdd
// 从文件系统中加载数据创建RDD
scala> val lines = sc.textFile("file:///usr/local/spark/mycode/rdd/word.txt")
lines: org.apache.spark.rdd.RDD[String] = file:///usr/local/spark/mycode/rdd/word.txt MapPartitionsRDD[12] at textFile at <console>:27
// 通过并行集合（数组）创建RDD
scala>val array = Array(1,2,3,4,5)
array: Array[Int] = Array(1, 2, 3, 4, 5)
scala>val rdd = sc.parallelize(array)
rdd: org.apache.spark.rdd.RDD[Int] = ParallelCollectionRDD[13] at parallelize at <console>:29

// RDD被创建好以后，在后续使用过程中一般会发生两种操作：
// *  转换（Transformation）： 基于现有的数据集创建一个新的数据集。
// *  行动（Action）：在数据集上进行运算，返回计算值。

// 转换操作
// * filter(func)：筛选出满足函数func的元素，并返回一个新的数据集
// * map(func)：将每个元素传递到函数func中，并将结果返回为一个新的数据集
// * flatMap(func)：与map()相似，但每个输入元素都可以映射到0或多个输出结果
// * groupByKey()：应用于(K,V)键值对的数据集时，返回一个新的(K, Iterable)形式的数据集
// * reduceByKey(func)：应用于(K,V)键值对的数据集时，返回一个新的(K, V)形式的数据集，其中的每个值是将每个key传递到函数func中进行聚合

// 行动操作
// * count() 返回数据集中的元素个数
// * collect() 以数组的形式返回数据集中的所有元素
// * first() 返回数据集中的第一个元素
// * take(n) 以数组的形式返回数据集中的前n个元素
// * reduce(func) 通过函数func（输入两个参数并返回一个值）聚合数据集中的元素
// * foreach(func) 将数据集中的每个元素传递到函数func中运行*

scala> val lines = sc.textFile("file:///usr/local/spark/mycode/rdd/word.txt")
lines: org.apache.spark.rdd.RDD[String] = file:///usr/local/spark/mycode/rdd/word.txt MapPartitionsRDD[16] at textFile at <console>:27
scala> lines.filter(line => line.contains("Spark")).count()
res1: Long = 2  //这是执行返回的结果

scala> val lines = sc.textFile("file:///usr/local/spark/mycode/rdd/word.txt")
scala> lines.map(line => line.split(" ").size).reduce((a,b) => if (a>b) a else b)

// 键值对RDD
scala> val list = List("Hadoop","Spark","Hive","Spark")
list: List[String] = List(Hadoop, Spark, Hive, Spark)
 
scala> val rdd = sc.parallelize(list)
rdd: org.apache.spark.rdd.RDD[String] = ParallelCollectionRDD[11] at parallelize at <console>:29
 
scala> val pairRDD = rdd.map(word => (word,1))
pairRDD: org.apache.spark.rdd.RDD[(String, Int)] = MapPartitionsRDD[12] at map at <console>:31
 
scala> pairRDD.foreach(println)
(Hadoop,1)
(Spark,1)
(Hive,1)
(Spark,1)

// 常用的键值对转换操作包括reduceByKey()、groupByKey()、sortByKey()、join()、cogroup()等，下面我们通过实例来介绍。
// reduceByKey(func)
scala> pairRDD.reduceByKey((a,b)=>a+b).foreach(println)
(Spark,2)
(Hive,1)
(Hadoop,1)

// groupByKey()
scala> pairRDD.groupByKey()
res15: org.apache.spark.rdd.RDD[(String, Iterable[Int])] = ShuffledRDD[15] at groupByKey at <console>:34
//从上面执行结果信息中可以看出，分组后，value被保存到Iterable[Int]中
scala> pairRDD.groupByKey().foreach(println)
(Spark,CompactBuffer(1, 1))
(Hive,CompactBuffer(1))
(Hadoop,CompactBuffer(1))

// keys
scala> pairRDD.keys
res17: org.apache.spark.rdd.RDD[String] = MapPartitionsRDD[17] at keys at <console>:34
scala> pairRDD.keys.foreach(println)
Hadoop
Spark
Hive
Spark

// values
scala> pairRDD.values
res0: org.apache.spark.rdd.RDD[Int] = MapPartitionsRDD[2] at values at <console>:34
 
scala> pairRDD.values.foreach(println)
1
1
1
1

// sortByKey()
scala> pairRDD.sortByKey()
res0: org.apache.spark.rdd.RDD[(String, Int)] = ShuffledRDD[2] at sortByKey at <console>:34
scala> pairRDD.sortByKey().foreach(println)
(Hadoop,1)
(Hive,1)
(Spark,1)
(Spark,1)

// mapValues(func)
scala> pairRDD.mapValues(x => x+1)
res2: org.apache.spark.rdd.RDD[(String, Int)] = MapPartitionsRDD[4] at mapValues at <console>:34
scala> pairRDD.mapValues(x => x+1).foreach(println)
(Hadoop,2)
(Spark,2)
(Hive,2)
(Spark,2)

// join
scala> val pairRDD1 = sc.parallelize(Array(("spark",1),("spark",2),("hadoop",3),("hadoop",5)))
pairRDD1: org.apache.spark.rdd.RDD[(String, Int)] = ParallelCollectionRDD[24] at parallelize at <console>:27
 
scala> val pairRDD2 = sc.parallelize(Array(("spark","fast")))
pairRDD2: org.apache.spark.rdd.RDD[(String, String)] = ParallelCollectionRDD[25] at parallelize at <console>:27
 
scala> pairRDD1.join(pairRDD2)
res9: org.apache.spark.rdd.RDD[(String, (Int, String))] = MapPartitionsRDD[28] at join at <console>:32
 
scala> pairRDD1.join(pairRDD2).foreach(println)
(spark,(1,fast))
(spark,(2,fast))

scala> val rdd = sc.parallelize(Array(("spark",2),("hadoop",6),("hadoop",4),("spark",6)))
rdd: org.apache.spark.rdd.RDD[(String, Int)] = ParallelCollectionRDD[38] at parallelize at <console>:27
 
scala> rdd.mapValues(x => (x,1)).reduceByKey((x,y) => (x._1+y._1,x._2 + y._2)).mapValues(x => (x._1 / x._2)).collect()
res22: Array[(String, Int)] = Array((spark,4), (hadoop,5))

// 共享变量
// 广播变量用来把变量在所有节点的内存之间进行共享。累加器则支持在所有不同节点之间进行累加计算（比如计数或者求和）
scala> val broadcastVar = sc.broadcast(Array(1, 2, 3))
broadcastVar: org.apache.spark.broadcast.Broadcast[Array[Int]] = Broadcast(0)
scala> broadcastVar.value
res0: Array[Int] = Array(1, 2, 3)
// 这个广播变量被创建以后，那么在集群中的任何函数中，都应该使用广播变量broadcastVar的值，而不是使用v的值，这样就不会把v重复分发到这些节点上。
// 此外，一旦广播变量创建后，普通变量v的值就不能再发生修改，从而确保所有节点都获得这个广播变量的相同的值。


// 第4章 Spark SQL
// RDD是分布式的 Java对象的集合，比如，RDD[Person]是以Person为类型参数，但是，Person类的内部结构对于RDD而言却是不可知的。
// DataFrame是一种以RDD为基础的分布式数据集，也就是分布式的Row对象的集合（每个Row对象代表一行记录），提供了详细的结构信息，也就是我们经常说的模式（schema），Spark SQL可以清楚地知道该数据集中包含哪些列、每列的名称和类型。

// 从Spark2.0以上版本开始，Spark使用全新的SparkSession接口替代Spark1.6中的SQLContext及HiveContext接口来实现其对数据加载、转换、处理等功能。
// SparkSession实现了SQLContext及HiveContext所有功能
scala> import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SparkSession
 
scala> val spark=SparkSession.builder().getOrCreate()
spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@2bdab835
 
//使支持RDDs转换为DataFrames及后续sql操作
scala> import spark.implicits._
import spark.implicits._
 
scala> val df = spark.read.json("file:///usr/local/spark/examples/src/main/resources/people.json")
df: org.apache.spark.sql.DataFrame = [age: bigint, name: string]
 
scala> df.show()
+----+-------+
| age|   name|
+----+-------+
|null|Michael|
|  30|   Andy|
|  19| Justin|
+----+-------+
 
 // 打印模式信息
scala> df.printSchema()
root
 |-- age: long (nullable = true)
 |-- name: string (nullable = true)
 
// 选择多列
scala> df.select(df("name"),df("age")+1).show()
+-------+---------+
|   name|(age + 1)|
+-------+---------+
|Michael|     null|
|   Andy|       31|
| Justin|       20|
+-------+---------+
 
// 条件过滤
scala> df.filter(df("age") > 20 ).show()
+---+----+
|age|name|
+---+----+
| 30|Andy|
+---+----+
 
// 分组聚合
scala> df.groupBy("age").count().show()
+----+-----+
| age|count|
+----+-----+
|  19|    1|
|null|    1|
|  30|    1|
+----+-----+
 
// 排序
scala> df.sort(df("age").desc).show()
+----+-------+
| age|   name|
+----+-------+
|  30|   Andy|
|  19| Justin|
|null|Michael|
+----+-------+
 
//多列排序
scala> df.sort(df("age").desc, df("name").asc).show()
+----+-------+
| age|   name|
+----+-------+
|  30|   Andy|
|  19| Justin|
|null|Michael|
+----+-------+
 
//对列进行重命名
scala> df.select(df("name").as("username"),df("age")).show()
+--------+----+
|username| age|
+--------+----+
| Michael|null|
|    Andy|  30|
|  Justin|  19|
+--------+----+
 
//  从RDD转换得到DataFrame

// 利用反射机制推断RDD模式
scala> import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
 
scala> import org.apache.spark.sql.Encoder
import org.apache.spark.sql.Encoder
 
scala> import spark.implicits._  //导入包，支持把一个RDD隐式转换为一个DataFrame
import spark.implicits._
 
scala> case class Person(name: String, age: Long)  //定义一个case class
defined class Person
 
scala> val peopleDF = spark.sparkContext.textFile("file:///usr/local/spark/examples/src/main/resources/people.txt").map(_.split(",")).map(attributes => Person(attributes(0), attributes(1).trim.toInt)).toDF()
peopleDF: org.apache.spark.sql.DataFrame = [name: string, age: bigint]
 
scala> peopleDF.createOrReplaceTempView("people")  //必须注册为临时表才能供下面的查询使用
 
scala> val personsRDD = spark.sql("select name,age from people where age > 20")
//最终生成一个DataFrame
personsRDD: org.apache.spark.sql.DataFrame = [name: string, age: bigint]
scala> personsRDD.map(t => "Name:"+t(0)+","+"Age:"+t(1)).show()  //DataFrame中的每个元素都是一行记录，包含name和age两个字段，分别用t(0)和t(1)来获取值
 
+------------------+
|             value|
+------------------+
|Name:Michael,Age:29|
|   Name:Andy,Age:30|
+------------------+


// 使用编程方式定义RDD模式
scala> import org.apache.spark.sql.types._
import org.apache.spark.sql.types._
 
scala> import org.apache.spark.sql.Row
import org.apache.spark.sql.Row
 
//生成 RDD
scala> val peopleRDD = spark.sparkContext.textFile("file:///usr/local/spark/examples/src/main/resources/people.txt")
peopleRDD: org.apache.spark.rdd.RDD[String] = file:///usr/local/spark/examples/src/main/resources/people.txt MapPartitionsRDD[1] at textFile at <console>:26
 
//定义一个模式字符串
scala> val schemaString = "name age"
schemaString: String = name age
 
//根据模式字符串生成模式
scala> val fields = schemaString.split(" ").map(fieldName => StructField(fieldName, StringType, nullable = true))
fields: Array[org.apache.spark.sql.types.StructField] = Array(StructField(name,StringType,true), StructField(age,StringType,true))
 
scala> val schema = StructType(fields)
schema: org.apache.spark.sql.types.StructType = StructType(StructField(name,StringType,true), StructField(age,StringType,true))
//从上面信息可以看出，schema描述了模式信息，模式中包含name和age两个字段
 
//对peopleRDD 这个RDD中的每一行元素都进行解析val peopleDF = spark.read.format("json").load("examples/src/main/resources/people.json")
 
 
scala> val rowRDD = peopleRDD.map(_.split(",")).map(attributes => Row(attributes(0), attributes(1).trim))
rowRDD: org.apache.spark.rdd.RDD[org.apache.spark.sql.Row] = MapPartitionsRDD[3] at map at <console>:29
 
scala> val peopleDF = spark.createDataFrame(rowRDD, schema)
peopleDF: org.apache.spark.sql.DataFrame = [name: string, age: string]
 
//必须注册为临时表才能供下面查询使用
scala> peopleDF.createOrReplaceTempView("people")
 
scala> val results = spark.sql("SELECT name,age FROM people")
results: org.apache.spark.sql.DataFrame = [name: string, age: string]
 
scala> results.map(attributes => "name: " + attributes(0)+","+"age:"+attributes(1)).show()
+--------------------+
|               value|
+--------------------+
|name: Michael,age:29|
|   name: Andy,age:30|
| name: Justin,age:19|
+--------------------+
 