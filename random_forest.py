import numpy as np

def load_data(filename):
    fr=open(filename)
    dataset=[]
    for line in fr.readlines():
        linearr=[]
        #1.提取每一行的元素可以先根据符号去除，然后再strip，或者如果直接是\t的话可以直接去除
        for element in line.split(","):
            element.strip()
            #2.判断不同数据类型再做不同处理，都是字符串要先转化为数据或者标签才能进一步处理
            if element.isdigit():
                linearr.append(float(element))
            else:
                linearr.append(element)
        dataset.append(linearr)
    return dataset

def split_left_and_right(index,value,dataset):
    #1.分割的办法
    left=list()
    right=list()
    for row in dataset:
        if row[index]<value:
            left.append(row)
        else:
            right.append(row)
    #2.返回的是一个元组，如果只用一个变量接受的话
    return left,right

def gini_impurity(groups,class_values):
    gini=0.0
    total=len(groups(0))+len(groups(1))
    #1.对每一个特征，计算在每一个组中的纯度
    for each_value in class_values:
        for group in groups:
            size=len(group)
            if size==0:
                continue
            #2.获取一个表中符合某个特征的人的总数,然后进一步得到如何获取比例
            proportion=[row[-1] for row in group].count(each_value)/float(size)
            gini+=(float(size)/total)*proportion*(1-proportion)
    return gini

def get_split(dataset,n_features):
    class_values=list(set([row[-1] for row in dataset]))
    b_index,b_value,b_score,b_groups=999,999,999,None#因为b_groups是元组
    features=list()
    for i in range(n_features):
        index=np.random.randint(0,len(dataset[0],1))
        if index not in features:
            features.append(index)
    for index in features:
        for row in dataset:
            groups=split_left_and_right(index,row[index],dataset)
            gini=gini_impurity(groups,class_values)
            if gini<b_score:
                b_index,b_value,b_score,b_groups=index,row[index],gini,groups
    return {"index":b_index,"value":b_value,"groups":b_groups}


