import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(filename):
    #1.读取时指定列名
    #2.要使用drop来删除特定列
    df=pd.read_csv(filename,sep=',',header=None,names=["Sex","Length","Diam","Height","Whole","Shucked","Viscera","Shell","Rings"
])
    df.drop(columns=["Rings"],axis=1,inplace=True)
    #3.采用独热编码
    #4.将独热编码产生的boolean变量类型转化为int型才方便计算
    modified_df=pd.get_dummies(df,columns=["Sex"])
    modified_df[["Sex_F","Sex_I","Sex_M"]]=modified_df[["Sex_F","Sex_I","Sex_M"]].astype(int)
    #5.df.head(num)显示前几列，用于调试的多
    #print(modified_df.head())

    fr=open(filename)
    labels=[]
    for line in fr.readlines():
        line.replace("\n","")
        elements=line.split(",")
        labels.append(float(elements[-1]))

    return modified_df,labels

def transform_data(df):
    array=df.to_numpy()
    return array

def ridge_regression(xMat,yMat,lamda):
    xTx=xMat.T*xMat
    demo=xTx+np.eye(xMat.shape[1])*lamda
    if np.linalg.det(demo) == 0:
        print("wrong! the det is zero")
        return
    print(demo.I.shape)
    print(xMat.T.shape)
    print(yMat.shape)
    ws=demo.I*xMat.T*yMat
    return ws

def ridge_test(xarray,yarray):
    xMat=np.mat(xarray)
    yMat=np.mat(yarray).T
    yMean=np.mean(yMat,0)
    #1.对yMat中心化
    yMat=yMat-yMean
    xMean=np.mean(xMat,0)
    xVar=np.var(xMat,0)
    #2.标准化xMat
    xMat=(xMat-xMean)/xVar

    numtest=30
    wMat=np.zeros((numtest,(xarray.shape)[1]))
    for i in range(numtest):
        ws=ridge_regression(xMat,yMat,np.exp(i-10))
        wMat[i,:]=ws.T
    return wMat


if __name__ =="__main__":
    df,labels=load_data('abalone.data')
    df_array=transform_data(df)
    res_weights=ridge_test(df_array,labels)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(res_weights)
    plt.show()