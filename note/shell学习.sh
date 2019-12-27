
# 引用学习
################################################################################################################################
# 引用var变量的值
$var 
${var}

# ${ }的模式匹配功能：
# #是去掉左边(在键盘上#在$之左边)
# %是去掉右边(在键盘上%在$之右边)
# #和%中的单一符号是最小匹配，两个相同符号是最大匹配。

# 引用var变量的值并且截取a左边的字符串显示，如果字符串有多个a，则按从左向右最后一个a截取
${var%a*} 

# 引用var变量的值并且截取a左边的字符串显示，如果字符串有多个a，则按从左向右第一个a截取
${var%%a*} 

# 引用var变量的值并且截取a右边的字符串显示，如果字符串有多个a，则按从左向右第一个a截取
${var#*a}

# 引用var变量的值并且截取a右边的字符串显示，如果字符串有多个a，则按从左向右最后一个a截取
${var##*a}

# 引用var变量的值并且从第3个字符开始截取
${var:3}

# 引用var变量的值并且从第3个字符开始截取，截取6个字符显示
${var:3:6}

# 返回var变量值的长度
${#var} 

# 替换变量值的字符串，这里是把var变量值里的a换成b,如果有多个a也只能换一次
${var/a/b}

# 替换变量值的字符串，这里把var变量值里的a全换成b
${var//a/b}

# eg:
var=123abcd456abcd

echo $var 
var=123abcd456abcd
echo ${var}
var=123abcd456abcd

echo  ${var%a*} 
123abcd456

echo  ${var%%a*} 
123

echo  ${var#*a}
bcd456abcd

echo  ${var##*a}
bcd

echo  ${var:3}
abcd456abcd

echo  ${var:3:6}
abcd45

echo  ${#var} 
14

echo  ${var/a/b}
123bbcd456abcd

echo  ${var//a/b}
123bbcd456bbcd

################################################################################################################################



# 括号的作用
################################################################################################################################
# 1
# []和test
# 两者是一样的，在命令行里test expr和[ expr ]的效果相同,三个基本作用是判断文件、判断字符串、判断整数
# 字符串比较：可用的比较运算符只有==和!=
# 整数比较：表达式需要写么写 -eq(对应==)，-gt(对应>)，-ge(对应>=)，-lt(对应<)，-le(对应<=)
# 无论是字符串比较还是整数比较都千万不要使用大于号小于号
a=aaa 
b=bbb

if [ $a = $b ]
then
   echo "$a = $b : TRUE"
else
   echo "$a = $b : FALSE"
fi

aaa = bbb : FALSE

if [ $a > $b ]
then
   echo "$a > $b : TRUE"
else
   echo "$a > $b : FALSE"
fi

aaa > bbb : TRUE

if [ $a \> $b ]
then
   echo "$a > $b : TRUE"
else
   echo "$a > $b : FALSE"
fi

aaa > bbb : FALSE

a=11
b=13

if [ $a > $b ]
then
   echo "$a > $b : TRUE"
else
   echo "$a > $b : FALSE"
fi

11 > 13 : TRUE

if [ $a -gt $b ]
then
   echo "$a > $b : TRUE"
else
   echo "$a > $b : FALSE"
fi

11 > 13 : FALSE


# 2
# [[]]
# 这是内置在shell中的一个命令，它就比刚才说的test强大的多了。支持字符串的模式匹配（使用=~操作符时甚至支持shell的正则表达式）。逻辑组合可以不使用test的-a,-o而使用&& ||。
# 字符串比较时可以把右边的作为一个模式（这是右边的字符串不加双引号的情况下。如果右边的字符串加了双引号，则认为是一个文本字符串。），而不仅仅是一个字符串，比如[[ hello == hell? ]]，结果为真。
a=aaa 
b=bbb

if [[ $a > $b ]]
then
   echo "$a > $b : TRUE"
else
   echo "$a > $b : FALSE"
fi

aaa > bbb : FALSE


a=11
b=13

if [[ $a > $b ]]
then
   echo "$a > $b : TRUE"
else
   echo "$a > $b : FALSE"
fi

11 > 13 : FALSE


# 正则测试
# a=nnn

# if [[ $a =~ ^a*  ]]
# then
#    echo "$a =~ ^a*  : TRUE"
# else
#    echo "$a =~ ^a*  : FALSE"
# fi


# 3
# (())和let
# 使用 (( )) 时，不需要空格分隔各值和运算符，使用 [[ ]] 时需要用空格分隔各值和运算符。
# (())和let两者也是一样的(或者说基本上是一样的，双括号比let稍弱一些)。主要进行算术运算(上面的两个都不行)，也比较适合进行整数比较，可以直接使用熟悉的<,>等比较运算符。
# 可以直接使用变量名如var而不需要$var这样的形式。支持分号隔开的多个表达式
#在(())中写表达式可以直接写 == ，>，>=，<，<= 

a=ccc
b=bbb

if (( $a > $b ))
then
   echo "$a > $b : TRUE"
else
   echo "$a > $b : FALSE"
fi

aaa > bbb : FALSE


a=11 
b=22

if (( $a > $b ))
then
   echo "$a > $b : TRUE"
else
   echo "$a > $b : FALSE"
fi

11 > 22 : FALSE

# 总结
# 在bash中，数字的比较最好使用 (( ))，虽说可以使用 [[ ]]，但若在其内使用运算符 >、>=、<、<=、==、!= 时，其结果经常是错误的
# 不过若在 [[ ]] 中使用 [ ] 中的运算符“-eq、-ne、-le、-lt、-gt、-ge”等，还尚未发现有错。
# 因此诸如$ [[ " a" != “b” && 4 > 3 ]] 这类组合（见上）也不可以在bash中使用，其出错率很高。

# $[]和$(())是一样的，都是进行数学运算的。支持+ - * / %（“加、减、乘、除、取模”）。但是注意，bash只能作整数运算，对于浮点数是当作字符串处理的。
# (( ))及[[ ]] 分别是[ ]的针对数学比较表达式和字符串表达式的加强版。其中[[ ]]中增加模式匹配特效 (( ))不需要再将表达式里面的大小于符号转义

################################################################################################################################


# 运算符学习
################################################################################################################################
# 原生bash不支持简单的数学运算，但是可以通过其他命令来实现，例如 awk 和 expr，expr 最常用。
# 表达式和运算符之间要有空格，例如 2+2 是不对的，必须写成 2 + 2，这与我们熟悉的大多数编程语言不一样。
# 算术运算符
echo `expr 2+2`
2+2
val=`expr 2+2`
echo $val
2+2
echo `expr 2 + 2`
4

a=10
b=20

echo `expr $a + $b`
echo `expr $a - $b`

# 乘号(*)前边必须加反斜杠(\)才能实现乘法运算；
echo `expr $a \* $b`

echo `expr $a / $b`
echo `expr $b / $a`

echo `expr $a % $b`
echo `expr $b % $a`

if [ $a == $b ]
then
   echo "a 等于 b"
fi


# 字符串拼接
dt1='20190101'
dtt1="${dt1:0:4}"".""${dt1:4:2}"".""${dt1:6:2}"".""00"

# 字符串运算符
a="abc"
b="efg"

# 跟算术运算符不同
if [ $a = $b ]
then
   echo "$a = $b : a 等于 b"
else
   echo "$a = $b: a 不等于 b"
fi

if [ $a != $b ]
then
   echo "$a != $b : a 不等于 b"
else
   echo "$a != $b: a 等于 b"
fi

if [ -z $a ]
then
   echo "-z $a : 字符串长度为 0"
else
   echo "-z $a : 字符串长度不为 0"
fi


if [ -n "$a" ]
then
   echo "-n $a : 字符串长度不为 0"
else
   echo "-n $a : 字符串长度为 0"
fi


if [ $a ]
then
   echo "$a : 字符串不为空"
else
   echo "$a : 字符串为空"
fi
################################################################################################################################



# 引用学习
################################################################################################################################
#shell强弱引用的区别
#shell中”“和'都可以用来做引用，当引用对象不是一个变量时，两种表达式没有太大区别，
#但是当引用对象是一个变量时，”“即弱引用，引用的是变量值，而'即强引用，引用的是变量本身的值。
name=loyal-Wang
echo "$name"
loyal-Wang
echo '$name'
$name
echo "'$name'"
'loyal-Wang'
echo '"$name"'
"$name"



# 原样输出字符串，不进行转义或取变量(用单引号)
echo '$name\"'

                   能否引用变量  |          能否引用转移符      |  能否引用文本格式符(如：换行符、制表符)

单引号  |           否           |             否             |            否

双引号  |           能           |             能             |            能

无引号  |           能           |             能             |            否
################################################################################################################################



# 循环学习
################################################################################################################################
# if else-if else
a=10
b=20
if [ $a == $b ]
then
   echo "a 等于 b"
elif [ $a -gt $b ]
then
   echo "a 大于 b"
elif [ $a -lt $b ]
then
   echo "a 小于 b"
else
   echo "没有符合的条件"
fi

# for 循环
for loop in 1 2 3 4 5
do
    echo "The value is: $loop"
done

for str in 'This is a string' 
do
    echo $str
done


for str in 'This is a string' 'This'
do
    echo $str
done

# while 语句
# 原生bash不支持简单的数学运算，但是可以通过其他命令来实现，例如 awk 和 expr，expr 最常用。
# let 命令是 BASH 中用于计算的工具，用于执行一个或多个表达式，变量计算中不需要加上 $ 来表示变量。如果表达式中包含了空格或其他特殊字符，则必须引起来。
# let 不需要空格隔开表达式的各个字符。而 expr 后面的字符需要空格隔开各个字符。
int=1
while (( $int<=5 ))
do
    echo $int
    # int=$(expr $int + 1)
    let int=$int+1
done
################################################################################################################################



# 函数学习
################################################################################################################################
# 无return
demoFun(){
    echo "这是我的第一个 shell 函数!"
}

demoFun

# 带有return语句
funWithReturn(){
    echo "这个函数会对输入的两个数字进行相加运算..."
    echo "输入第一个数字: "
    read aNum
    echo "输入第二个数字: "
    read bNum
    echo "两个数字分别为 $aNum 和 $bNum !"
    return $(($aNum+$bNum))
}

funWithReturn

echo "输入的两个数字之和为 $? !"

# 函数参数
funWithParam(){
    echo "第一个参数为 $1 !"
    echo "第二个参数为 $2 !"
    echo "第十个参数为 $10 !"
    echo "第十个参数为 ${10} !"
    echo "第十一个参数为 ${11} !"
    echo "参数总数有 $# 个!"
    echo "$*"
    echo "$@"
}
funWithParam 1 2 3 4 5 6 7 8 9 34 73
################################################################################################################################


# linux中sed命令用法
# sed是一个很好的文件处理工具，本身是一个管道命令，主要是以行为单位进行处理，可以将数据行进行替换、删除、新增、选取等特定工作，sed命令行格式为：
# sed [-nefri] ‘command’ 输入文本

# sed -e 命令：直接在指令列模式上进行 sed 的动作编辑，只是将替换结果打印到屏幕上
# sed -i 命令：保存结果到文件中


# 一、基本的替换
# 命令格式1：sed 's/原字符串/新字符串/' 文件
# 命令格式2：sed 's/原字符串/新字符串/g' 文件
# 没有“g”表示只替换第一个匹配到的字符串（每行），有“g”表示替换所有能匹配到的字符串，“g”可以认为是“global”（全局的）的缩写，没有“全局的”结尾就不要替换全部

# cat aaa.txt
h4_1
h4_2
h4_3
h4_4
h4_5
h4_6

# sed 's/4/t/' aaa.txt
ht_1
ht_2
ht_3
ht_4
ht_5
ht_6

# sed 's/4/t/g' aaa.txt
ht_1
ht_2
ht_3
ht_t
ht_5
ht_6


# 二、替换某行内容
# 命令格式1：sed '行号c 新字符串' 文件

# 命令格式2：sed '起始行号，终止行号c 新字符串' 文件 多行替换成一行

# sed '2c test' aaa.txt

h4_1
test
h4_3
h4_4
h4_5
h4_6

# sed '2,6c test' aaa.txt

h4_1
test

# 循环实现每行替代
# 不可行
for ((i=2;i<=6;i++));
do 
sed '"${i}"c test' aaa.txt
done 


# 三、多条件替换
# 命令格式：sed -e 命令1 -e 命令2 -e 命令3

# sed -e 's/4/t/g' -e '2c test' aaa.txt 

ht_1
test
ht_3
ht_t
ht_5
ht_6


# sed '$c test' aaa.txt  $表示最后一行

# 批量更改文件名
# 查询测试代码
for i in `ls //home/sdk_analyst/hejinyang/temp`
do
    k=`echo $i | grep '20180801'`
    t=`echo $i | grep 'hjump2'`
    name1=`echo $k | sed 's/_20180801_/_20180802_/g'` 
    echo $k $name1
    name2=`echo $t | sed 's/hjump2_/hjump1_/g'` 
    echo $t $name2
done


# 实际代码
# 方法1：rename函数
rename 's/20180801/20180802/g' *.csv
rename 's/csv/csv.del/g' *.csv


# 方法2：sed函数
# 思考怎么使日期自动增加方法
date=$(date +%Y%m%d)
date1=$(date -d "+1 day" +%Y%m%d)
for i in `ls //home/sdk_analyst/hejinyang/temp`
do
    name1=`echo $i | grep "$date"`
    name2=`echo $name1 | sed 's/"$date"/"$date1"/g'` 
    echo $name1 $name2
done

sed -i -e "s/^/h4_/" xxx.csv

# 循环学习
################################################################################################################################
# while循环
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


# if循环
res1="a"
if [ $res1 == "a" ]; then
    echo "aaa"
elif [ $res1 == "d" ]; then
    echo "bbb"
else
    echo "ccc"
fi


# for循环


# for跟if嵌套循环
for i in `ls /home/sdk_analyst/hejinyang/hejy_file`
do
    k=`echo $i | grep '_list_1_'`
    t=`echo $i | grep '_list_2_'`
    if [ ! -z $k ];then
        name1=`echo $k | sed 's/_list_1_/_list_3_/g'` 
        echo $k $name1
    elif [ ! -z $t ];then
        name2=`echo $t | sed 's/_list_2_/_list_4_/g'` 
        echo $t $name2
    else
        echo "no files"
    fi
done

for i in `hdfs dfs -ls /user/hive/warehouse/dw_stage_open/dw_stage_open.db/hejy_push/`
do
    name1=`echo $i | grep "csv"`
	name2="$i"".del"
    if [ ! -z $name1 ]
	then hdfs dfs -mv $name1 $name2
	fi
done

# while跟if嵌套循环
res1=""
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


start_day=20180501
end_day=20190504

insert_day=$start_day

while [[ $insert_day -le $end_day ]]
do
echo ${insert_day}
insert_day=$(date -d "+1 day $insert_day " +%Y%m%d)
done
################################################################################################################################





# 数组学习
################################################################################################################################
# Shell通过特定字符把字符串分割成数组
str="1,2,3,4";
# //与/之间与分割的字符 ，另外/后有一个空格不可省略
str=${str//,/ };
# 转化为数组
arr=($str);

# shell 数组
my_array=(A B "C" D)
echo $my_array[0]
A[0]
echo ${my_array[0]}
A


echo ${my_array[1]}
B
echo ${my_array[*]}
A B C D
echo ${#my_array[1]}
1
echo ${#my_array[*]}
4
my_array[0]=aaa
echo ${my_array[*]}
aaa B C D
# 数组长度
echo ${#my_array[*]}
4
# 元素长度
echo ${#my_array[1]}
1
echo ${#my_array[0]}
3


#遍历数组
for each in ${arr[*]}
do
 echo $each
done

# 分片访问形式为：${数组名[@或*]:开始下标:结束下标}，注意，不包括结束下标元素的值。
${arr_number[@]:1:4}

# 实战：取出三个partition中的一个
k=$(/app/hadoop/hive/bin/hive -e "show partitions sdk_user.hejy_push_list_all partition (pt='2018-08-01');")
str=${k//game_name=/ };
arr=($str);

# 取出奇数的元素
for(( i=1;i<${#arr[@]};i+=2 )) do 
echo ${arr[i]}; 
done;


# 加载不定量的数组为固定格式
count=4
res1="aaa bbb ccc ddd"
t=$(expr $count - 1)
my_array=($res1)

echo $t
echo ${my_array[0]}
echo ${my_array[*]}

k="('XGG0001','YX0001"
for((i=0;i<=$t;i++))
do
	echo ${my_array[$i]}
	k="$k""','""${my_array[$i]}"
	echo $k
	if [[ $i == $t ]]; then
	k="$k""')"
	echo $k
fi
done

push_result_rid(){
	start_time=$(date +%s)
    exec 2>&1 >> ${log_dir}/${dt0}_hejy_push_result_rid.log 

    count=`${hive_cmd}/hive -e "
    select count(1) 
    from dw.wifi_app_batch_message_request_day_his
    where pt='$dt1'
    and to_date(last_updated_dt)>='$dt4'
    and app_id='A0008';
    "`
    
    rid=`${hive_cmd}/hive -e "
    select bat_msg_req_id 
    from dw.wifi_app_batch_message_request_day_his
    where pt='$dt1'
    and to_date(last_updated_dt)>='$dt4'
    and app_id='A0008';
    "`
    
    t=$(expr $count - 1)
    array=($rid)

    k="('XGG0001','YX0001"
    for((i=0;i<=$t;i++))
    do
    	echo ${array[$i]}
    	k="$k""','""${array[$i]}"
    	echo $k
    	if [[ $i == $t ]]; then
    	k="$k""')"
    	echo $k
    fi
    done

    end_time=$(date +%s)
    cost_time=$(expr ${end_time} / 60 - ${start_time} / 60)
    echo "***********************push_result_rid's cost_time is ${cost_time} minutes***********************"
}
################################################################################################################################


# 输出多个字符
str=$(printf "%-200s" "-")
echo "${str// /-}" 
--------------------------


# 对文件排序
假设当前已有文件system.txt，内容如下：其中空白部分为单个制表符。

[root@linuxidc tmp]# cat system.txt
1       mac     2000    500
2       winxp   4000    300
3       bsd     1000    600
4       linux   1000    200
5       SUSE    4000    300
6       Debian  600     200
(1).不加任何选项时，将对整行从第一个字符开始依次向后直到行尾按照默认的字符集排序规则做升序排序。
[root@linuxidc tmp]# sort system.txt

(2).以第三列为排序列进行排序。由于要划分字段，所以指定字段分隔符。指定制表符这种无法直接输入的特殊字符的方式是$'\t'。
[root@linuxidc tmp]# sort -t $'\t' -k3 system.txt  

(3).对第三列按数值排序规则进行排序。
[root@linuxidc tmp]# sort -t $'\t' -k3 -n system.txt

(4).在对第3列按数值排序规则排序的基础上，使用第四列作为决胜属性，且是以数值排序规则对第四列排序。
[root@linuxidc tmp]# sort -t $'\t' -k3 -k4 -n system.txt