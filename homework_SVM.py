# -*- coding: utf-8 -*-
#Author: Fan Mao Ting
#
import numpy as np
import cvxopt
import matplotlib.pyplot as pl

#读取数据
def loadDataSet(filename):
    dataMat=[]
    labelMat=[]
    fr=open(filename)
    for lineArr in fr:
        lineArr=lineArr.split(' ')
        for i in range(len(lineArr)):
            lineArr[i]=float(lineArr[i])
        dataMat.append(lineArr[0:-1])
        labelMat.append(lineArr[-1])
    fr.close()
    dataMat=np.array(dataMat)
    labelMat=np.array(labelMat)
    #将标签设置为1或-1
    i=0
    m,n=np.shape(dataMat)
    while (i<m):
        if labelMat[i]!=1 :
            labelMat[i]=-1
        else:
            labelMat[i]=1
        i += 1
    return dataMat,labelMat


#用cvxopt进行优化
def Cvx(X,y,C):
    m,n=np.shape(y)
    XX=np.dot(X,X.T)
    YY=np.dot(y,y.T)
    P=cvxopt.matrix(XX*YY)
    q=cvxopt.matrix(-1*np.ones(m))
    G1=np.eye(m)
    G2=np.diag(np.ones(m)*-1)
    G= cvxopt.matrix(np.vstack((G1,G2)))
    h1=np.ones(m)*C
    h2=np.zeros(m)
    h= cvxopt.matrix(np.hstack((h1,h2)))
    A= cvxopt.matrix(y.T)
    b= cvxopt.matrix(0.0)
    solve= cvxopt.solvers.qp(P,q,G,h,A,b)
    return np.array(solve['x'])

#计算w,b
def calculate_w_b(X,y,a):
    Index=np.where(a>1e-9)
    Alpha= a[Index]
    Label= y[Index]
    Data = X[Index[0],:]
    w=np.dot(Data.T,Label*Alpha)
    b=np.mean(Label-np.dot(Data,w))
    return w,b
#计算精度
def calculate_accuracy(X,y,w,b):
    bingo=0
    m,n=np.shape(y)
    F=np.dot(X,w)+b
    for i in range(m):
        if np.sign(F[i])==y[i]:
            bingo +=1
    accuracy=bingo/m
    return accuracy
#Leave-One-Out验证
def DataMat_split(X,y):
    Size=54
    Data=[]
    Label=[]
    for i in range(5):
        Data.append(X[Size*i:Size*(i+1)])
        Label.append(y[Size*i:Size*(i+1)])
    return Data,Label
def Train_Test(DataMatgroup,LabelMatgroup,k):
    Data_Train=np.vstack(DataMatgroup[0:k]+DataMatgroup[k+1:])
    Label_Train=np.vstack(LabelMatgroup[0:k]+LabelMatgroup[k+1:])
    Data_Test=DataMatgroup[k]
    Label_Test=LabelMatgroup[k]
    return Data_Train,Label_Train,Data_Test,Label_Test


def main():
    filename_data='/Users/fmtaxz/PycharmProjects/untitled2/traindata.csv'
    X,y=loadDataSet(filename_data)
    y=y.reshape(-1,1)
    Data,Label=DataMat_split(X,y)
    C_list=[]
    accuracy_list=[]
    c=0.0001
    while c<=1000000:
        C_list.append(c)
        c=c*10
    optimal_accuracy,optimal_C=0,0
    for i in C_list:
        accuracy_list1=[]
        for k in range(5):
            Data_Train, Label_Train, Data_Test, Label_Test=Train_Test(Data,Label,k)
            a=Cvx(Data_Train,Label_Train,i)
            w,b=calculate_w_b(Data_Train,Label_Train,a)
            accuracy=calculate_accuracy(Data_Test,Label_Test,w,b)
            accuracy_list1.append(accuracy)

        mean_accuracy=np.mean(accuracy_list1)
        accuracy_list.append(optimal_accuracy)

        if mean_accuracy>optimal_accuracy:
            optimal_accuracy,optimal_C=mean_accuracy,i
    print('Optimal accuracy is: %f\nOptimal C is: %f '%(optimal_accuracy,optimal_C))
    pl.figure()
    pl.title("Curve of accuracy")
    pl.plot(accuracy_list[0:],'b--o',label='accuracy')
    pl.legend()
    pl.xlabel('time')
    pl.ylabel('accuracy')
    pl.grid()
    pl.show()


if __name__=='__main__':
    main()



