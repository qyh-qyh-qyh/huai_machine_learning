
import numpy as np
def load_data():
    '''
    :description:此次为自建数据集，加载数据集成为类似张量的格式
    :return: 返回加载好的数据集和相应标签
    '''
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # [0,0,1,1,1......]
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec

def create_vocablist(dataset):
    '''
    :description:创建一个包含所有单词的单词表
    :param dataset: 已经转换为张量形式的文章单词
    :return: 返回一个包含所有单词的单词向量
    '''
    vocablist=set([])
    for words in dataset:
        #1.几个集合相加还维持集合的用并集
        vocablist=vocablist | set(words)
    #print(np.shape(list(vocablist)))#正确的size为32
    return list(vocablist)

def words_to_vec(vocablist,input_words):
    '''
    :description:根据所有词向量将输入句子中含有的单词的位置置为1
    :param vocablist: 单词向量表
    :param input_words: 输入的一段文字，要经过准备数据阶段
    :return: 返回一个向量标识input_words这句话含有什么单词
    '''
    return_vec=[0 for i in range(len(vocablist))]
    #不遍历单词表应该更快
    for word in input_words:
        #如果在单词表中的话可以提供判断信息
        if word in vocablist:
            #1.获取索引位置
            return_vec[vocablist.index(word)]=1
    return return_vec

def train_navie_bayes(train_matrix,train_labels):
    '''
    :description:计算权重参数即分子上的概率
    :param train_matrix: 多条训练语句的向量的集合
    :param train_labels: 训练语句的类别的归类
    :return: 返回几个参数，用概率向量表示
    '''
    #1.为了防止出现0导致乘法积直接为0的情况出现
    p0_vocablist=np.array([1.0 for i in range(len(train_matrix[0]))])
    #print(np.shape(p0_vocablist))
    p1_vocablist = np.array([1.0 for i in range(len(train_matrix[0]))])
    #计算类别为0，1时每个词出现的总个数,起始为2.0(乱设置的，其实主要是为了比较大小)
    p0_times=2.0
    p1_times=2.0
    p0_frequence=0.0
    for train_words in train_matrix:
        if train_labels[train_matrix.index(train_words)]==1:
            #3.转为ndarray才是广播相加不然是直接加再末尾,debug发现
            p1_vocablist+=train_words
            #print(np.shape(p1_vocablist))
            p1_times+=sum(train_words)
        else:
            p0_vocablist+=train_words
            p0_times+=sum(train_words)
            p0_frequence+=1.0
    #2.采取log计算怕计算结果太小，其实感觉也没有必要
    p0_vec=np.log(np.array(p0_vocablist)/p0_times)
    #print(np.shape(p0_vocablist))
    p1_vec=np.log(np.array(p1_vocablist)/p1_times)
    p0_prob=p0_frequence/len(train_matrix)
    #print('p0_prob=',p0_prob)
    return p0_vec,p1_vec,p0_prob

def classify_navie_bayes(vec_to_classify,p0_vec,p1_vec,p0_prob):
    #print(np.shape(p0_vec))
    #1.用当前向量和权重相乘，可以理解为加权，看当前属于哪个类别概率多大
    #print(vec_to_classify)
    #print(p0_vec)
    #print(np.log(p0_prob))
    p0=np.sum(vec_to_classify * p0_vec)+np.log(p0_prob)
    p1=np.sum(vec_to_classify * p1_vec)+np.log(1-p0_prob)
    if p1>p0:
        return 1
    return 0

def test_naive_bayes():
    dataset,train_labels=load_data();
    vocablist=create_vocablist(dataset)
    train_matrix=[]
    for data in dataset:
        data_to_vec=words_to_vec(vocablist,data)
        train_matrix.append(data_to_vec)
    p0_vec,p1_vec,p0_prob=train_navie_bayes(train_matrix,train_labels)
    test_words=["love","my","dalmation"]
    test_vec=words_to_vec(vocablist,test_words)
    #print(np.shape(test_vec))
    print(classify_navie_bayes(test_vec,p0_vec,p1_vec,p0_prob))
    test_words = ["stupid", "garbage"]
    test_vec = words_to_vec(vocablist,test_words)
    print(classify_navie_bayes(test_vec, p0_vec, p1_vec, p0_prob))

if __name__=="__main__":
    test_naive_bayes()
