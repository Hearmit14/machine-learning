val spark=SparkSession.builder().enableHiveSupport().getOrCreate()
 
val sc = ss.sparkContext


val M_rdd= sc.parallelize(
    Array(
        Array(0, 0, 1),
        Array(0, 1, 2),
        Array(0, 2, 3),
        Array(1, 0, 4),
        Array(1, 1, 5),
        Array(1, 2, 6)
    )
)

val N_rdd= sc.parallelize(
    Array(
        Array(0, 0, 7),
        Array(0, 1, 8),
        Array(1, 0, 9),
        Array(1, 1, 10),
        Array(2, 0, 11),
        Array(2, 1, 12)
    )
)
//M_rdd = sc.parallelize([(0, 0, 1), (0, 1, 2), (0, 2, 3), (1, 0, 4), (1, 1, 5), (1, 2, 6)])
//N_rdd = sc.parallelize([(0, 0, 7), (0, 1, 8), (1, 0, 9), (1, 1, 10), (2, 0, 11), (2, 1, 12)])

import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.mllib.linalg.Vectors

//val M_matrix = M_rdd.map(f => Vectors.dense(
//f(0).toDouble,
//f(1).toDouble,
//f(2).toDouble))

//val N_matrix = N_rdd.map(f => Vectors.dense(
//f(0).toDouble,
//f(1).toDouble,
//f(2).toDouble))

val M_matrix = M_rdd.map(f => (
f(0).toLong,
f(1).toLong,
f(2).toDouble))


val M = M_matrix.map(f => new MatrixEntry(f(0),f(1),f(2)))

val ent1 = new MatrixEntry(0,1,0.5)
val ent2 = new MatrixEntry(2,2,1.8)
val entries = sc.parallelize(Array(ent1,ent2))

val M = new CoordinateMatrix(M_matrix)

val N = new CoordinateMatrix(N_matrix)
 
M = M.entries.map(lambda entry: (entry.j, (entry.i, entry.value)))
N = N.entries.map(lambda entry: (entry.i, (entry.j, entry.value)))
 
matrix_entries = M.join(N).values().map(
    lambda x: ((x[0][0], x[1][0]), x[0][1] * x[1][1])
).reduceByKey(
    lambda x, y: x + y
).map(
    lambda x: MatrixEntry(x[0][0], x[0][1], x[1])
)
 
matrix = CoordinateMatrix(matrix_entries)
matrix.entries.collect()
 
######## 输出 #############
# [MatrixEntry(0, 0, 58.0),
#  MatrixEntry(1, 0, 139.0),
#  MatrixEntry(0, 1, 64.0),
#  MatrixEntry(1, 1, 154.0)]


    val rdd1= sc.parallelize(
      Array(
        Array(1.0,7.0,0,0),
        Array(0,2.0,8.0,0),
        Array(5.0,0,3.0,9.0),
        Array(0,6.0,0,4.0)
      )
    ).map(f => Vectors.dense(f))
