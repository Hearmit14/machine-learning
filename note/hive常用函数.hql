--windows文件转换为linux
find . -type f -exec dos2unix {} +


-- group by造成数据倾斜
set hive.map.aggr=true;--在map中会做部分聚集操作，效率更高但需要更多的内存
SET hive.groupby.skewindata=true;--两个mr job，第一个MR job中，Map的输出结果集合会随机分布到Reduce中

-- 不同数据类型关联产生数据倾斜
-- 解决方法：把数字类型转换成字符串类型
on a.usr_id = cast(b.user_id as string)

-- 空key造成数据倾斜
-- 赋随机值
ON CASE WHEN a.id IS NULL THEN concat('hive', rand()) ELSE a.id END = b.id;

-- 某些特殊key值引起的数据倾斜
set hive.optimize.skewjoin = true;
set hive.skewjoin.key = 1000;

--hive设置语句
set mapred.reduce.tasks=1000;
set mapred.reduce.tasks=1000;

set mapreduce.job.queuename=root.yarn_analyst.analyst

--hive助手上传数据：utf-8文本格式

--与oracle不同，hive的instr不能指定起始位置与查找的个数
--常用函数
--截取函数
--无中文时
substr(device_id,1,instr(device_id,'-')-1)
--截取‘-’前一段数据
--有中文时
regexp_extract(device_id,'^.([0-9]+[a-z]+)-(.*)$',1)
regexp_extract(device_id,'^.(^-)+(.*)$',1)
--有分隔符的字符串分隔
--'.'
split(num,'\\|')[0] as index_1,split(num,'\\|')[1] as name,split(num,'\\|')[2] as model
--split用多个分割符
split(name,"[-—@{}<>=:：《()（）]")[0]
split(name,"-|@|{|}|<|>|=|:")[0]

--列转行
--oracle
select t.rank, WMSYS.WM_CONCAT(t.Name) TIME From t_menu_item t GROUP BY t.rank;
--hive
select ubi.id,ubi.name,concat_ws(',', collect_set(ubi.address)) as address from user_basic_info ubi group by ubi.id,ubi.name;


--行转列
select col1,col2,name from test lateral view explode(split(col3,',')) col3 as name;


--日期处理
select substr(insert_time,1,instr(insert_time,' ')-1) from hejy_xz_game_active_1;

select from_unixtime(unix_timestamp(insert_time,'yyyy/MM/dd h:mm'),'yyyy/mm/dd') from temp.hejy_xz_game_active_1;

select device_id,to_char(to_date(insert_time,'yyyy-mm-dd hh24-mi-ss'),'yyyy-mm-dd') as date1,data_date as date2 from hejy_xz_game_active_1;

select date_sub(to_date(from_unixtime(unix_timestamp(insert_time,'yyyy/MM/dd h:mm'),'yyyy-MM-dd HH:mm:ss')),1) as date1,data_date from hejy_xz_game_reg_2;


--cts时间处理
and from_unixtime(cast(substr(cts,1,10) as int),'yyyy-MM-dd') = pt  --是否当天


--查看列名跟数据
set hive.cli.print.header=true;  // 打印列名 
set hive.cli.print.row.to.vertical=true;   // 开启行转列功能, 前提必须开启打印列名功能 
set hive.cli.print.row.to.vertical.num=1; // 设置每行显示的列数 


--查看表更新时间
desc formatted


--建表语句
drop table if exists sdk_user.hejy_model_base_data;
create table sdk_user.hejy_model_base_data
(
dhid                    string,
aid                     string,
imei                    string,
mac                     string,
reg_days                int,
province                string,
city                    string,
manuf                   string,
app_list                string,
new_funid_list          string,
flow_pkg_list           string
)
partitioned by(pt string)
row format delimited
fields terminated by '\t'
ESCAPED BY '\\' 
STORED AS orc;

--导入数据
load data local inpath '/home/sdk_analyst/hejinyang/hejy_temp_pk_0505.txt' overwrite into table sdk_user.hejy_indiana_push_list partition(pt='2017-05-15');

insert overwrite table sdk_user.wifi_push_api partition(pt='2017-07-21')
select 22000004 as subject_Id,dhid from 
(select dhid,prob
from sdk_user.hejy_landlord_bundle_model_9
order by prob desc
limit 50000000)a;

--删除分区数据
ALTER TABLE sdk_user.hejy_push_list_all DROP IF EXISTS PARTITION (pt='2018-06-15',flag_owner=1,game_name='buyu_1');

--增加分区数据
ALTER TABLE sdk_user.hejy_push_list_all add partition(pt=3) partition(pt=4);

--更改表名
ALTER TABLE name RENAME TO new_name;

--增加字段
ALTER TABLE employee ADD COLUMNS (dept STRING COMMENT 'Department name');

--更改列名
ALTER TABLE name CHANGE column_name new_name new_type

--${hiveconf:subject_id}无引号
drop table if exists sdk_user.hejy_temp_model_3_${hiveconf:subject_id};

--随机按比例抽样取数
--把随机的行数取模
drop table if exists sdk_user.hejy_model_build_5_3_click;
create table sdk_user.hejy_model_build_5_3_click as
select *,pmod(rn2,10) as flag_rand from
(select *,row_number() over(partition by flag_model order by rand(111)) as rn2 from
sdk_user.hejy_model_build_5_2_click)b;



--查看分区是否存在，等待分区数据补完
res1=""
while true;
do
  res1=`/app/hadoop/hive/bin/hive -e "show partitions dw_coredb.wifi_usertrackinfo_D_incr partition (pt='$dt1')"`
  if [[ ! -z $res1 ]]; then
    res1=`/app/hadoop/hive/bin/hive -e "show partitions ods_coredb.wifi_userdevice_H_incr partition (pt='$dt1')"`
    if [[ ! -z $res1 ]]; then
        echo "all partitions are ready, going to escape"
        break;
    fi
  fi
  echo "not all partitions are ready, wait..."
  sleep 10
done


--查看表内的最新一个分区
pt1=$(/app/hadoop/hive/bin/hive -e  "show partitions dw.wifi_userappinfo_device_app_list;"|sort | tail -n 1)
eval $pt1


-- cd /mnt/mfs/biz_data_gz/dhid2subject

-- cd /home/sdk_analyst/hejinyang/hejy_log

-- cd /data/app/push/csv_list/

-- cd /data/app/push/cur_model

-- cd /data/app/push/cur_version

-- cd /data/mfsjh/kfpt/guanggao300/subject_plan_idea/
-- cd /data/mfsjh/kfpt/guanggao300/num_inview_click/
