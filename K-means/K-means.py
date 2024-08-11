import time
import xlrd
import xlwt
import numpy as np

def load_data(filename):
    #1.读取一个工作簿
    workbook=xlrd.open_workbook(filename)
    #2.读取工作簿中的工作表
    dealsheet=workbook.sheet_by_index(0)
    score_list=[]
    #3.获取行数的方法
    for i in range(1,dealsheet.nrows):
        #4.获取特定值的办法
        if dealsheet.cell_value(i,4)!="":
            score_list.append(dealsheet.cell_value(i,4))
    return score_list

def distEclud(score1,score2):
    #print(score1,score2)
    #1.计算两点之间距离公式
    return np.sqrt(np.power(score1-score2,2))

def rand_cent(data_list,k):
    centroids=np.mat(np.zeros((k,1)))
    min_num=np.min(data_list)
    range_num=float(np.max(data_list)-np.min(data_list))
    #2.生成特定的范围的值
    centroids[:,0]=min_num+range_num*np.random.rand(k,1)
    return centroids

def write_to_res_file(centroids,flag):
    with open("result.txt","a") as f:
        f.write(u"第%d次迭代" %flag)
        for i in range(centroids.shape[0]):
            res=''
            res+=str(float(centroids[i,0]))+" "
        f.write(res)
        f.write("\n"+"******************************"+"\n\n\n")

def score_kmeans(data_list,k,dist=distEclud,createCentroids=rand_cent):
    #表示经过第几次迭代
    flag=0
    #报错1：没有把data_list转化为mat或者ndarray导致用不了np处理
    #重点：目前已有数据和质心的数组，此时一个质心要对应一个数据，可以建立一个数组，该数组中索引对应一个个数据点，然后数组根据索引定位的值可以存储质心的值
    num=len(data_list)
    data_list=np.mat(data_list)
    data_list=data_list.reshape((data_list.shape[1],1))
    cluster_array=np.mat(np.zeros((num,2)))
    #print(cluster_array.shape)
    centroids=createCentroids(data_list,k)
    #print(centroids)
    #重点：判断是不是要继续可以单独采用一个变量,初始为true，进入就为false
    change_cluster=True
    while change_cluster:
        change_cluster=False
        flag+=1
        for i in range(num):
            #重点：要用一个数组建立联系的时候，一个值储存存进入数组的值，另一个储存判断是否更新的值
            min_index=-1
            min_value=np.inf
            for j in range(k):
                distance=dist(data_list[i,0],centroids[j,0])
                if distance<min_value:
                    min_index=j
                    min_value=distance
            #报错2：索引用[row,col]形式，不然报错是说一个array的比较，需要使用any或者all
            if min_index != cluster_array[i,0]:
                change_cluster=True
            #print(min_index,min_value)
            cluster_array[i,:]=np.array([min_index,min_value])
        for center in range(k):
            index_group=[x for x in range(num) if cluster_array[x,0]==center]
            all_current_center_cluster=np.array(data_list[index_group])
            centroids[center]=np.mean(all_current_center_cluster)
        write_to_res_file(centroids,flag)
    return centroids,cluster_array,flag

def write_data(cluster_list,my_center,flag,filename):
    start=65
    start_rank=cluster_list[0,0]
    workbook=xlrd.open_workbook(filename)
    dealsheet=workbook.sheet_by_index(0)
    #1.创建一个新的工作簿
    newexcel=xlwt.Workbook()
    #2.创建一个新的工作表
    newsheet=newexcel.add_sheet(u'sheet1',cell_overwrite_ok=True)
    for i in range(dealsheet.nrows):
        #3.写的时候用if排除掉缺失的数据，如果缺失数据不在最后应该采取其他办法
        if dealsheet.cell_value(i,4)!="":
            for j in range(dealsheet.ncols):
                #4.向新表中写入数据
                newsheet.write(i,j,dealsheet.cell_value(i,j))
            if i!=0:
                newsheet.write(i,dealsheet.ncols,cluster_list[i-1,0])
                if start_rank == cluster_list[i-1,0]:
                    newsheet.write(i,dealsheet.ncols+1,chr(start))
                else:
                    start_rank=cluster_list[i-1,0]
                    start+=1
                    newsheet.write(i,dealsheet.ncols+1,chr(start))
    newsheet.write(0,dealsheet.ncols,"cluster_number")
    newsheet.write(0,dealsheet.ncols+1,"level")
    newsheet.write(0,dealsheet.ncols+2,"centroids_middle")
    for i in range(1,len(my_center)+1):
        newsheet.write(i,dealsheet.ncols+2,my_center[i-1,0])
    newexcel.save("K-means.xls")

if __name__=="__main__":
    file_name="math2.xls"
    k=4
    score_list=load_data(file_name)
    centroids,cluster_array,flag=score_kmeans(score_list,k)
    with open ("K-means.txt","a") as f:
        #日志处理
        f.write(u"log"+str(time.time())+"\n")
    print(u"迭代了%d次"%flag)
    write_data(cluster_array,centroids,flag,file_name)




