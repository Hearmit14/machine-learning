from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import xlearn as xl

train = pd.read_csv(
    '/Users/hejinyang/Desktop/machine_learning/pratice/loan/train_ctrUa4K.csv')

test = pd.read_csv(
    '/Users/hejinyang/Desktop/machine_learning/pratice/loan/test_lAUu6dG.csv')


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

cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome',
        'CoapplicantIncome', 'LoanAmount', 'Loan_Status', 'Credit_History', 'Property_Area']

train_sub = train[cols]

train_sub['Gender'].fillna('Male', inplace=True)
train_sub['Married'].fillna('Yes', inplace=True)
train_sub['Dependents'].fillna('0', inplace=True)
train_sub['Self_Employed'].fillna('No', inplace=True)
train_sub['LoanAmount'].fillna(146.412162, inplace=True)
train_sub['Credit_History'].fillna(0, inplace=True)

train_sub.info()

dict_ls = {'Y': 1, 'N': 0}
train_sub['Loan_Status'].replace(dict_ls, inplace=True)

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

df = X_train
numerics = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
categories = ['Gender', 'Married', 'Dependents',
              'Education', 'Self_Employed', 'Loan_Status', 'Credit_History', 'Property_Area']
features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome',
            'CoapplicantIncome', 'LoanAmount', 'Credit_History', 'Property_Area']


def convert_to_ffm(df, type, numerics, categories, features):
    currentcode = len(numerics)
    catdict = {}
    catcodes = {}

    # Flagging categorical and numerical fields
    for x in numerics:
        catdict[x] = 0
    for x in categories:
        catdict[x] = 1

    nrows = df.shape[0]
    ncolumns = len(features)

    with open(str(type) + "_ffm.txt", "w") as text_file:

        # Looping over rows to convert each row to libffm format
        for n, r in enumerate(range(nrows)):
            datastring = ""
            datarow = df.iloc[r].to_dict()
            # Set Target Variable here
            datastring += str(int(datarow['Loan_Status']))

           # For numerical fields, we are creating a dummy field here
            for i, x in enumerate(catdict.keys()):
                if(catdict[x] == 0):
                    datastring = datastring + " " + \
                        str(i)+":" + str(i)+":" + str(datarow[x])
                else:

                   # For a new field appearing in a training example
                    if(x not in catcodes):
                        catcodes[x] = {}
                        currentcode += 1
                        # encoding the feature
                        catcodes[x][datarow[x]] = currentcode

           # For already encoded fields
                    elif(datarow[x] not in catcodes[x]):
                        currentcode += 1
                        # encoding the feature
                        catcodes[x][datarow[x]] = currentcode

                    code = catcodes[x][datarow[x]]
                    datastring = datastring + " " + \
                        str(i)+":" + str(int(code))+":1"

            datastring += '\n'
            text_file.write(datastring)
