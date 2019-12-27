import pandas as pd
import numpy as np
import xgboost as xgb
import re

train = pd.read_csv("/Users/hejinyang/Downloads/all/train.csv")
test  = pd.read_csv("/Users/hejinyang/Downloads/all/test.csv")

X_y_train = xgb.DMatrix(data=train[['Pclass', 'Age', 'Fare', 'SibSp', 'Parch']], 
                        label=train['Survived'])
X_test    = xgb.DMatrix(data=test[['Pclass', 'Age', 'Fare', 'SibSp', 'Parch']])

params = {
          'base_score': np.mean(train['Survived']),
          'eta':  0.1,
          'max_depth': 3,
          'gamma' :3,
          'objective'   :'reg:linear',
          'eval_metric' :'mae'
         }


model = xgb.train(params=params, 
                  dtrain=X_y_train, 
                  num_boost_round=3)


xgb.to_graphviz(booster = model, num_trees=0)
xgb.to_graphviz(booster = model, num_trees=1)
xgb.to_graphviz(booster = model, num_trees=2)

model.get_dump()

#a.split('\n')

#第一棵树转为矩阵
model.get_dump()[0].split('\n')

#第一棵树的第一组参数(矩阵)
model.get_dump()[0].split('\n')[0]

#第一棵树的第一组参数转为字符串
s="".join(model.get_dump()[0].split('\n')[0])



tree_num=0
node_index = 0
node_index = 7

#遍历一颗树
def string_parser(tree_num,node_index,unique_id,table_in):
    all_tree = model.get_dump()[tree_num].split('\n')
    all_tree1 = "".join(model.get_dump()[tree_num].split('\n'))
    layer_index = re.findall(r"(\d+):", all_tree1)

    cur_tree = "".join(all_tree[node_index])
    cur_layer = re.findall(r"[\t]+", cur_tree)
    node_type = len(re.findall(r":leaf=", cur_tree))

    if node_type == 0:
        out  = re.findall(r"[\w.-]+", cur_tree)
        if len(cur_layer) == 0:        # 根节点没有括号
            return ('select ' + unique_id +',\ncase when ' + out[1] + '<' + out[2] +' then ' + 
                    "".join(string_parser(tree_num,layer_index.index(out[4]),unique_id,table_in)) + 
                    '\n'+'when ' + out[1] + '>=' + out[2] +' then ' + 
                    "".join(string_parser(tree_num,layer_index.index(out[6]),unique_id,table_in)) + 
                    '\n'+'when ' + out[1] + ' is null' + ' then ' + 
                    "".join(string_parser(tree_num,layer_index.index(out[8]),unique_id,table_in)) + 
                    '\nend as pred \nfrom ' + table_in)
        else:
            return ('\n'+'(case when ' + out[1] + '<' + out[2] +' then ' + 
                    "".join(string_parser(tree_num,layer_index.index(out[4]),unique_id,table_in)) + 
                    '\n'+'when ' + out[1] + '>=' + out[2] +' then ' + 
                    "".join(string_parser(tree_num,layer_index.index(out[6]),unique_id,table_in)) + 
                    '\n'+'when ' + out[1] + ' is null' + ' then ' + 
                    "".join(string_parser(tree_num,layer_index.index(out[8]),unique_id,table_in)) + 
                    '\n'+'end)' )
    else:
        out = re.findall(r"[\d.-]+", cur_tree)
        return (out[1])


#string_parser(0,0,unique_id="dhid",table_in="hejy_temp_1")


#增加每棵树之间的代码细节
#return只输出一次，到输出时在循环
def tree_parser(tree,tree_num,unique_id,table_in,table_out):
    if tree_num == 0:
        return ('drop table if exists ' + table_out + ';\ncreate table ' + table_out + ' stored as orc as' + 
                '\nselect ' +  unique_id + ' , 1/(1+exp(-( 0 + sum(pred)))) AS pred \nfrom (\n'  + 
                 "".join(string_parser(tree_num,0,unique_id,table_in)))
    elif tree_num == len(tree)-1:
        return ('\nunion all\n' + "".join(string_parser(tree_num,0,unique_id,table_in)) +
                '\n)a\ngroup by ' + unique_id + ';')
    else:
        return ('\nunion all\n' + "".join(string_parser(tree_num,0,unique_id,table_in)))
        


#遍历每颗树
#tree=model.get_dump()
def gbdt_to_sql(tree,unique_id,table_in,table_out,sql_out_dir):
    sql = ''
    for tree_num in range(len(tree)):
        sql += tree_parser(tree,tree_num,unique_id,table_in,table_out)
    
    if sql_out_dir[-1] == '/':
        sql_out = sql_out_dir + 'gbdt_to_sql.sql'
    else:
        sql_out = sql_out_dir + '/gbdt_to_sql.sql'

    fo = open(sql_out,'w+')
    fo.write(sql)
    fo.close


gbdt_to_sql(model.get_dump(),"dhid","hejy_temp_1","hejy_temp_2","/Users/hejinyang/Desktop")
