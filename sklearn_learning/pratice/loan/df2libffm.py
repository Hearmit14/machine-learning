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
