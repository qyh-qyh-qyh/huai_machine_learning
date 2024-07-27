import operator

import numpy as np
#1.收集数据略

#2.准备数据：从文件中取出数据并变成想要的形式
def file_to_matrix(filename):
    '''
    desc:
        导入训练数据
    param filename:
        读取的目标文件
    return:
        返回特征的矩阵和标签列表
    '''
    #1.打开文件并采用readlines读取文件
    fr=open(filename)

    #2.获取文件的行数和列数
    data_row_num=len(fr.readlines())
    #确保文件不为空
    if(data_row_num==0):
        return None
    fr=open(filename)
    data_col_num=len(fr.readline().split("\t"))-1#-1是除去lable
    #3.利用获得的行数和列数构造一个初始化的返回矩阵
    return_matrix=np.zeros((data_row_num,data_col_num))
    return_labels=np.zeros((data_row_num))
    idx=0
    fr=open(filename)
    #4.索引加文件逐行读取逐行修改返回值
    for line in fr.readlines():
        line=line.strip()
        line_list=line.split("\t")
        return_matrix[idx, :] = line_list[0:data_col_num]
        return_labels[idx]=line_list[-1]
        idx+=1
    return return_matrix,return_labels

#3.处理数据，处理异常值，缺失值，以及画图或者打印，归一化等等此处采用归一化
def autoform(dataset):
    #采用X-Xmin/Xmax-Xmin归一化到0-1区间
    #1.注意利用哪个做轴相当于把当前的数据集投影到那个轴上，然后再处理
    min_val=dataset.min(axis=0)
    max_val=dataset.max(axis=0)
    #2.通过数据集获取数据集行数的方法
    dataset_row_num=dataset.shape[0]
    #3.利用数据集shape初始化目标矩阵
    normal_matrix=np.zeros(np.shape(dataset))
    ranges=max_val-min_val
    #4.利用np.tile将数组扩展
    normal_matrix=dataset-np.tile(min_val,(dataset_row_num,1))
    normal_matrix=normal_matrix/np.tile(ranges,(dataset_row_num,1))
    return normal_matrix,ranges,min_val

def classify(inx,dataset,labels,k):
    '''
    :desc:
        采用k近邻计算可能的标签
    :param inx:
        经过归一化处理的一组特征值
    :param dataset:
        归一化完成的数据集
    :param labels:
        每个数据集的标签
    :param k:
        KNN中的参数K
    :return:
        返回预测的标签
    '''
    #1.有小维度和大维度数据时，不一定是把大的数据一个个拆解，如果操作一样也可以用小的维度扩展
    dataset_row_num=dataset.shape[0]
    extend_inx_to_matrix=np.tile(inx,(dataset_row_num,1))
    diff=extend_inx_to_matrix-dataset
    diff_squared=diff**2
    sum_distance=diff_squared.sum(axis=1)
    distance=sum_distance**0.5
    #2.给排序并返回索引值的列表，一般用在通过一个值排序好但是要从中获取的数据不在原有排序好的数据集里的情况，这时通过索引能够从别的数据集访问到数据
    sort_distance=distance.argsort()
    class_count={}
    for i in range(k):
        vote=labels[sort_distance[i]]
        class_count[vote]=class_count.get(vote,0)+1
    #3.字典的排序操作和上一行的默认值处理方法
    sort_class_count=sorted(class_count.items(),key=operator.itemgetter(1),reverse=True)
    return sort_class_count[0][0]

def class_test(radio):
    data_read,label_read=file_to_matrix("datingTestSet2.txt")
    data_normal,ranges,minval=autoform(data_read)
    data_row_num=data_normal.shape[0]
    test_num=int(radio*data_row_num)
    error_num=0.0
    for i in range(test_num):
        classify_res=classify(data_normal[i,:],data_normal[test_num:data_row_num],label_read[test_num:data_row_num],3)
        if(classify_res!=label_read[i]):
            error_num+=1.0
    error_rate=error_num/test_num
    print("The error rate is: ",error_rate)
    return error_rate

if __name__=="__main__":
    radio=float(input("Please input the radio of the test set: "))
    error_rate=class_test(radio)

