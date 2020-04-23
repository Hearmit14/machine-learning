from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import xlearn as xl

train = pd.read_csv(
    '/Users/hejinyang/Desktop/machine_learning/pratice/loan/train_ctrUa4K.csv')

test = pd.read_csv(
    '/Users/hejinyang/Desktop/machine_learning/pratice/loan/test_lAUu6dG.csv')

# 查看数据
train.head()
train.iloc[0]

# 显示索引与列名
train.index
train.columns

train.info()
# Loan_ID              614 non-null object
# Gender               601 non-null object
# Married              611 non-null object
# Dependents           599 non-null object
# Education            614 non-null object
# Self_Employed        582 non-null object
# ApplicantIncome      614 non-null int64
# CoapplicantIncome    614 non-null float64
# LoanAmount           592 non-null float64
# Loan_Amount_Term     600 non-null float64
# Credit_History       564 non-null float64
# Property_Area        614 non-null object
# Loan_Status 614 non - null object

train.describe()
# ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  Credit_History
# count       614.000000         614.000000  592.000000         600.00000      564.000000
# mean       5403.459283        1621.245798  146.412162         342.00000        0.842199
# std        6109.041673        2926.248369   85.587325          65.12041        0.364878
# min         150.000000           0.000000    9.000000          12.00000        0.000000
# 25 % 2877.500000           0.000000  100.000000         360.00000        1.000000
# 50 % 3812.500000        1188.500000  128.000000         360.00000        1.000000
# 75 % 5795.000000        2297.250000  168.000000         360.00000        1.000000
# max       81000.000000       41667.000000  700.000000         480.00000        1.000000

train['Self_Employed'].value_counts()

train['Gender'].fillna('Male', inplace=True)
train['Married'].fillna('Yes', inplace=True)
train['Dependents'].fillna('0', inplace=True)
train['Self_Employed'].fillna('No', inplace=True)
train['LoanAmount'].fillna(146.412162, inplace=True)
train['Credit_History'].fillna(0, inplace=True)

dict_ls = {'Y': 1, 'N': 0}
train['Loan_Status'].replace(dict_ls, inplace=True)

cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome',
        'CoapplicantIncome', 'LoanAmount', 'Loan_Status', 'Credit_History', 'Property_Area']

train_sub = train[cols]


X_train, X_test = train_test_split(train_sub, test_size=0.3, random_state=5)

# vec = DictVectorizer()
# X_train = vec.fit_transform(train_sub.to_dict(orient='record'))
# vec.feature_names_
# print(X_train[0])
# print(len(vec.feature_names_))
# X_train.shape
# (614, 21)

# df = Dataframe to be converted to ffm format
# Type = ‘Train’ / ‘Test’/ ‘Valid’
# Numerics = list of all numeric fields
# Categories = list of all categorical fields
# Features = list of all features except the Label and Id

numerics = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
categories = ['Gender', 'Married', 'Dependents',
              'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
# features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome',
#             'CoapplicantIncome', 'LoanAmount', 'Credit_History', 'Property_Area']

x_train = X_train[['Loan_Status']+numerics+categories]
x_test = X_test[['Loan_Status']+numerics+categories]


# 特征分类
feas_n = numerics
feas_c = categories
feas_oh = []


class df2libffm:
    def __init__(self, feas_n, feas_c, feas_oh):
        self.catdict = {}
        for x in feas_n:
            self.catdict[x] = 0  # 数值型特征
        for x in feas_c:
            self.catdict[x] = 1  # 类别单值型特征
        for x in feas_oh:
            self.catdict[x] = 2  # one-hot后的类别多值型特征
        self.field_ids = {}
        self.feat_ids = {}
        self.fieldcode = 0
        self.featcode = 0
    # 初始化

    def build(self, df):
        for n, r in enumerate(range(len(df))):
            datarow = df.iloc[r].to_dict()
            for i, x in enumerate(self.catdict.keys()):
                # 数值型特征
                if(self.catdict[x] == 0):
                    if(x not in self.field_ids):
                        self.field_ids[x] = self.fieldcode
                        self.fieldcode += 1
                        self.feat_ids[x] = self.featcode
                        self.featcode += 1
                # 类别单值型特征
                if(self.catdict[x] == 1):
                    if(x not in self.field_ids):
                        self.field_ids[x] = self.fieldcode
                        self.fieldcode += 1
                        self.feat_ids[x] = {}
                        self.feat_ids[x][datarow[x]] = self.featcode
                        self.featcode += 1
                     # For already encoded fields
                    elif(datarow[x] not in self.feat_ids[x]):
                        self.feat_ids[x][datarow[x]] = self.featcode
                        self.featcode += 1
                # 类别多值型特征
                if(self.catdict[x] == 2):
                    if(x.split('_')[0] not in self.field_ids):
                        self.field_ids[x.split('_')[0]] = self.fieldcode
                        self.fieldcode += 1
                        self.feat_ids[x] = self.featcode
                        self.featcode += 1
    # 转换

    def gen(self, df, path, dtype):
        with open(path, "w") as text_file:
            for n, r in enumerate(range(len(df))):
                datastring = ""
                datarow = df.iloc[r].to_dict()
                # 第一列：target
                if dtype == 'train':
                    datastring += str(int(datarow['Loan_Status']))
                if dtype == 'valid':
                    datastring += str(int(datarow['Loan_Status']))
                if dtype == 'test':
                    datastring += str(int(0))
                # 第二列开始：特征编码
                for i, x in enumerate(self.catdict.keys()):
                    if(self.catdict[x] == 0):
                        datastring = datastring + " " + \
                            str(self.field_ids[x])+":" + \
                            str(self.feat_ids[x])+":" + str(str(datarow[x]))
                    if(self.catdict[x] == 1):
                        datastring = datastring + " " + \
                            str(self.field_ids[x])+":" + \
                            str(self.feat_ids[x][datarow[x]])+":1"
                    if(self.catdict[x] == 2):
                        if datarow[x] == 1:
                            datastring = datastring + " " + \
                                str(self.field_ids[x.split('_')[0]]) + \
                                ":" + str(self.feat_ids[x])+":1"
                datastring += '\n'
                text_file.write(datastring)


df_ffm = df2libffm(feas_n, feas_c, feas_oh)
df_ffm.build(x_test)

# df_ffm.field_ids
# df_ffm.feat_ids

path = '/Users/hejinyang/Desktop/machine_learning/pratice/loan/test.txt'
df_ffm.gen(x_test, path, 'valid')

ffm_model = xl.create_ffm()

ffm_model.setTrain(
    "/Users/hejinyang/Desktop/machine_learning/pratice/loan/train.txt")
ffm_model.setValidate(
    "/Users/hejinyang/Desktop/machine_learning/pratice/loan/test.txt")

param = {'task': 'binary',
         'lr': 0.1,
         'lambda': 0.002,
         'metric': 'auc',
         'fold': 3,
         'opt': 'adagrad'}

# ffm_model.setTXTModel("./model.txt")
ffm_model.fit(
    param, '/Users/hejinyang/Desktop/machine_learning/pratice/loan/model.out')


ffm_model.setSigmoid()
ffm_model.setTest(
    "/Users/hejinyang/Desktop/machine_learning/pratice/loan/small_test.txt")
ffm_model.predict("/Users/hejinyang/Desktop/machine_learning/pratice/loan/model.out",
                  "/Users/hejinyang/Desktop/machine_learning/pratice/loan/output.txt")
