import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    fr=open(filename)
    data_matrix=[]
    label_matrix=[]
    for line in fr.readlines():
        data_line=line.strip().split("\t")
        # split函数获取的都是字符串，所以要先转换值
        data_line=[float(element) for element in data_line]
        #1.这种矩阵的要加一个1.0在前面，并且除去最后一个元素加到label中
        data_matrix.append([1.0]+data_line[0:-1])#注意：extend不返回值，就地修改列表所以这里错误，应该用加法
        label_matrix.append([data_line[-1]])#2.从根源解决读数据的问题,标签读取要读取数组中的数组，才能变成（5，1）的shape
    return data_matrix,label_matrix

def sigmoid(inX):
    #1.函数中有一个参数也可以直接替换成向量的
    #print(inX.shape)
    #res=2*1.0/(1.0+np.exp(-2*inX))-1
    #print(res.shape)
    return 2*1.0/(1.0+np.exp(-2*inX))-1

def logistic_regression(data_matrix,label_matrix):
    #1.先把数据转化为矩阵才能用矩阵乘法,还有返回结果的mat和array相互转化
    data_numpy_matrix=np.array(data_matrix)
    label_numpy_matrix=np.array(label_matrix)
    #print(label_numpy_matrix.shape)
    #np.array才有shape属性，list没有
    #print(data_numpy_matrix.shape)
    m,n=data_numpy_matrix.shape
    weights=np.ones((n,1))
    #print(weights)
    #print(weights.shape)
    max_cycles=500
    alpha=0.001
    for i in range(max_cycles):
        calculate=sigmoid(np.dot(data_numpy_matrix,weights))
        #print(calculate.shape)
        error=label_numpy_matrix-calculate
        #print(label_numpy_matrix.shape)
        #print(error.shape)
        #不是很知道为什么用error,是因为要实时修改alpha还是凑啥
        weights=weights+alpha*np.dot(data_numpy_matrix.transpose(),error)
        #print(weights)
    return np.array(weights)
def plot_best_fit(data_matrix,label_matrix,weights):
    #获取列表维度的方法与shape比较
    n=len(data_matrix)
    #这里还要用隔开
    x0=[];y0=[]
    x1=[];y1=[]
    for i in range(n):
        if(label_matrix[i]==0):
            x0.append(data_matrix[i][0])
            y0.append(data_matrix[i][1])
        else:
            x1.append(data_matrix[i][0])
            y1.append(data_matrix[i][1])
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.scatter(x0,y0,s=8,c="r")
    ax.scatter(x1,y1,s=8,c="g")
    ax.set_title("Logistic Regression")
    ax.set_xlabel("data_matrix[0]")
    ax.set_ylabel("data_matrix[1]")
    x=np.arange(-3.0,3.0,0.1)
    #print(weights)
    #print(x.shape)
    y=(-weights[0]-np.dot(weights[1],x))/weights[2]
    ax.plot(x,y,'b^-',label="best_fit")
    ax.legend()
    plt.show()

def testLR():
    data_matrix,label_matrix=load_data("logistic_data.txt")
    weights=logistic_regression(data_matrix,label_matrix)
    plot_best_fit(data_matrix,label_matrix,weights)

if __name__=="__main__":
    testLR()



