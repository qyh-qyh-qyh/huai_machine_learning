#决策树的代码

import numpy as np
from collections import Counter

#1.收集数据，文本文件理解，学会了理解方法

#2.准备数据，读取数据并且以合适的格式储存
def read_data(filename):
    fr=open(filename)
    dataset=[line.strip().split("\t") for line in fr.readlines()]
    return dataset
#3.分析数据

#4.训练算法
def calculate_entropy(dataset):
    '''
    :desc:计算一个数据集的熵，只根据标签计算就好
    :param dataset: 输入的数据集，一定包含标签，因为要根据此进行计算
    :return: 返回计算的熵结果
    '''
    data_row_num=len(dataset)
    #1.采用字典可以计数每一个标签的个数,很适合要计算这种每个数都参与的运算
    labels={}
    for i in range(data_row_num):
        now_label=dataset[i][-1]
        labels[now_label]=labels.get(now_label,0)+1
    entropy=0.0
    for key in labels:
        prob=float(labels[key])/data_row_num
        entropy-=(prob*np.log(prob,2))
    return entropy

def split_data(dataset,idx,value):
    '''
    :desc:分割一组数据，提取出其中在某一特征上满足某一值的子数据集，进而继续在子数据集进行决策树分类
    :param dataset: 待分割的数据集
    :param idx: 分割点的特征
    :param value: 分割点特征的值
    :return: 返回分割好的子数据集,子数据集删除了符合分割点要求的数据
    '''
    #1.挑选符合要求的一组数据然后删除特定信息的办法
    return_dataset=[]
    for i in range(len(dataset)):
        if dataset[i][idx]==value:
            reduced_data=dataset[i][:idx]
            reduced_data=dataset[i][idx+1:]
            return_dataset.append(reduced_data)
    return reduced_data

def find_best_feature(dataset):
    #1.获取大小-》初始化答案-》计算
    feature_num=len(dataset[0])-1
    base_entropy=calculate_entropy(dataset)
    best_feature,best_entropy=-1,0.0
    for i in range(feature_num):
        #2.获取所有列的值并对每一列所有的值确保唯一之后处理
        now_feature_list=[each_data[i] for each_data in dataset]
        unique_feature_list=set(now_feature_list)
        new_entropy=0.0
        for each_feature in unique_feature_list:
            sub_dataset=split_data(dataset,i,each_feature)
            prob_each_feature=len(sub_dataset)/float(len(dataset))
            new_entropy+=prob_each_feature*calculate_entropy(sub_dataset)
        #到此获取了每一个特征值的加权后的熵值。，这个值要尽可能低这样表明分类是最有效的
        diff_entropy=base_entropy-new_entropy
        if diff_entropy>best_entropy:
            best_feature=i
            best_entropy=diff_entropy
    return best_feature



def create_tree(dataset,labels):
    label_list=[example[-1] for example in dataset]
    if len(set(label_list))==1:
        return label_list[0]
    if len(dataset[0])==1:
        return Counter(label_list)

    best_feature_idx=find_best_feature(dataset)
    best_feature_name=labels[best_feature_idx]

    mytree={best_feature_name:{}}
    del labels[best_feature_idx]
    best_feature_list=[example[best_feature_idx] for example in dataset]
    unique_best_feature_list=set(best_feature_list)
    for each_feature in unique_best_feature_list:
        sub_labels=labels[:]
        mytree[best_feature_name][each_feature]=create_tree(split_data(dataset,best_feature_idx,each_feature),sub_labels)
    return mytree

def classify(input_tree,feature_names,test_vector):