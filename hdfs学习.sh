--HDFS文件系统学习
--查看文件目录
show create table dw_coredb.wifi_usertrackinfo_d_incr;

--查看分区表入库信息
hadoop fs -ls  hdfs://ns1/user/hive/warehouse/dw.db/wifi_userappinfo_device_app_list/pt=2018-05-08

--查看文件
hdfs dfs -cat /user/hive/warehouse/sdk/sdk_user.db/hejy_temp_model_1/000343_0|head -10

--导出文件到本地
hdfs dfs -copyToLocal /user/hive/warehouse/sdk/sdk_user.db/hejy_temp_model_1/000343_0 /home/sdk_analyst/hejinyang/
--会导出所以目录及文件
hdfs dfs -copyToLocal /user/hive/warehouse/sdk/sdk_user.db/auto_push_hive_model_of* /home/sdk_analyst/hejinyang/

--本地文件上传
hdfs dfs -put /home/sdk_analyst/hejinyang/aaa.txt /user/hive/warehouse/dw_stage_open/dw_stage_open.db/hejy_push

--创建目录
hdfs dfs -mkdir /user/hive/warehouse/dw_stage_open/dw_stage_open.db/hejy_push

--删除文件
hdfs dfs -rm /user/hive/warehouse/dw_stage_open/dw_stage_open.db/hejy_push/*.del


# 金华hdfs文件系统数据处理
# 方法1
for i in `hdfs dfs -ls /user/hive/warehouse/dw_stage_open/dw_stage_open.db/hejy_push/`
do
    name1=`echo $i | grep "csv"`
	name2="$i"".del"
    if [ ! -z $name1 ]
	then hdfs dfs -mv $name1 $name2
	fi
done


# 方法2
t=($(hdfs dfs -ls /user/hive/warehouse/dw_stage_open/dw_stage_open.db/hejy_push/))
echo ${#t[*]}
i=1
y=${#t[*]}
while (( $i<=$y )) 
do 
    name1=`echo ${t[$i]} | grep "csv"`
	name2="${t[$i]}"".del"
    if [ ! -z $name1 ]
	then echo $name1 $name2
	fi
    let i=$i+1
done


# 不导出表直接从hive的文件系统读数据
hdfs dfs -cat /user/hive/warehouse/sdk/sdk_user.db/hejy_push_list_all_1/pt=2018-11-15/game_name=buyu_2/000000_0|head -10

# hive表对应的目录中的文件被替换之后，相应的表也会发生改变
drop table if exists test.hejy_temp_1 ;
CREATE TABLE test.hejy_temp_1
(
pkg string COMMENT '清理前包名',
name string COMMENT '清理后包名'
)
stored as orc;

show create table test.hejy_temp_1 ;

hadoop fs -ls hdfs://ShareSdkHadoop/user/hive/warehouse/dm_sdk_mapping.db/app_pkg_clean_byhand/000000_0

hdfs://ShareSdkHadoop/user/hive/warehouse/test.db/hejy_temp_1


hdfs dfs -cp /user/hive/warehouse/dm_sdk_mapping.db/app_pkg_clean_byhand/000000_0 /user/hive/warehouse/test.db/hejy_temp_1

