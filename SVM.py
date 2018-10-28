# -*- coding: utf-8 -*-
# Author: 范茂廷


import numpy as np
import cvxopt
from sklearn.model_selection import train_test_split
#读取数据
def loadDataSet(filename):
    dataMat=[]
    labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=line.strip().split(' ')
        dataMat.append([float(lineArr[0]),float(lineArr[1]),float(lineArr[2]),float(lineArr[3]),
                        float(lineArr[4]),float(lineArr[5]),float(lineArr[6]),float(lineArr[7]),
                        float(lineArr[8]),float(lineArr[9]),float(lineArr[10]),float(lineArr[11]),
                        float(lineArr[12])])
        labelMat.append(float(lineArr[13]))
    #将标签设置为1或-1
    i=0
    while (i<270):
        if labelMat[i]==2 :
            labelMat[i]=-1
        i += 1
    X_train, X_test, Y_train, Y_test =train_test_split(dataMat, labelMat, test_size=0.16667,random_state=40)  # 这里训练集5/6:测试集1/6.
    print(Y_test,Y_train)
    return X_train, X_test, Y_train, Y_test

def selectJrand(i,m): #在0-m中随机选择一个不是i的整数X_train, X_test, Y_train, Y_test
    j=i
    while (j==i):
        j=int(np.random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):  #保证a在L和H范围内（L <= a <= H）
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

def kernelTrans(X, A, kTup): #核函数，输入参数,X:支持向量；A：某一行特征数据；kTup：('lin',k1)核函数的类型和参数
    m,n = np.shape(X)
    K = np.mat(np.zeros((m,1)))
    if kTup[0]=='lin': #线性函数
        K = X * A.T+1
    elif kTup[0]=='rbf': # 径向基函数(radial bias function)
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = np.exp(K/(-1*kTup[1]**2)) #返回生成的结果
    else:
        raise NameError('Not support kernel type!')
    return K


#定义类，方便存储数据
class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # 存储各类参数
        self.X = dataMatIn  #数据特征
        self.labelMat = classLabels #数据类别
        self.C = C #软间隔参数C
        self.tol = toler #停止阀值
        self.m = np.shape(dataMatIn)[0] #数据行数
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0 #初始设为0
        self.eCache = np.mat(np.zeros((self.m,2))) #缓存
        self.K = np.mat(np.zeros((self.m,self.m))) #核函数的计算结果
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

def calcEk(oS, k): #计算Ek（参考《统计学习方法》p127公式7.105）
    fXk = float(np.multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

#随机选取aj，并返回其E值
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcacheList = np.nonzero(oS.eCache[:,0])[0]  #返回矩阵中的非零位置的行数
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE): #返回步长最大的aj
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej



def updateEk(oS, k): #更新os数据
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

#首先检验ai是否满足KKT条件，如果不满足，随机选择aj进行优化，更新ai,aj,b值
def innerL(i, oS): #输入参数i和所有参数数据
    Ei = calcEk(oS, i) #计算E值
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)): #检验这行数据是否符合KKT条件 参考《统计学习方法》p128公式7.111-113
        j,Ej = selectJ(i, oS, Ei) #随机选取aj，并返回其E值
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]): #以下代码的公式参考《统计学习方法》p126
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H:
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #参考《统计学习方法》p127公式7.107
        if eta >= 0:
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta #参考《统计学习方法》p127公式7.106
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L) #参考《统计学习方法》p127公式7.108

        if (abs(oS.alphas[j] - alphaJold) < oS.tol): #alpha变化大小阀值（自己设定）
            updateEk(oS, j)
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#参考《统计学习方法》p127公式7.109
        updateEk(oS, i) #更新数据
        #以下求解b的过程，参考《统计学习方法》p129公式7.114-7.116
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]<oS.C):
            oS.b = b1
        elif (0 < oS.alphas[j]<oS.C):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        updateEk(oS, i)
        updateEk(oS, j)
        return 1
    else:
        return 0


#SMO函数，用于快速求解出alpha
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)): #输入参数：数据特征，数据类别，参数C，阀值toler，最大迭代次数，核函数（默认线性核）
    oS = optStruct(np.mat(dataMatIn),np.mat(classLabels).T,C,toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m): #遍历所有数据
                alphaPairsChanged += innerL(i,oS)
                #print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)) #显示第多少次迭代，那行特征数据使alpha发生了改变，这次改变了多少次alpha
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs: #遍历非边界的数据
                alphaPairsChanged += innerL(i,oS)
                #print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        #print("iteration number: %d" % iter)
    return oS.b,oS.alphas

def testRbf(data_train):
    dataArr_train,dataArr_test,labelArr_train,labelArr_test = loadDataSet(data_train) #读取训练数据
    C=1
    x=0
    while(x<6):
        C=10*C
        x=x+1
        b,alphas = smoP(dataArr_train, labelArr_train, C, 0.00001, 10000, ('rbf',100)) #通过SMO算法得到b和alpha
        datMat=np.mat(dataArr_train)
        labelMat = np.mat(labelArr_train).T
        svInd= np.nonzero(alphas)[0]  #选取不为0数据的行数（也就是支持向量）
        sVs=datMat[svInd] #支持向量的特征数据
        labelSV = labelMat[svInd] #支持向量的类别（1或-1）
        print("there are %d Support Vectors" % np.shape(sVs)[0]) #打印出共有多少的支持向量
        m,n = np.shape(datMat) #训练数据的行列数
        errorCount = 0
        for i in range(m):
            kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', 100)) #将支持向量转化为核函数
            predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b  #这一行的预测结果（代码来源于《统计学习方法》p133里面最后用于预测的公式）注意最后确定的分离平面只有那些支持向量决定。
            if np.sign(predict)!= np.sign(labelArr_train[i]): #sign函数 -1 if x < 0, 0 if x==0, 1 if x > 0
                errorCount += 1
        print("When C = %f, the training error rate is: %f" % (C,float(errorCount)/m)) #打印出错误率
        errorCount_test = 0
        datMat_test= np.mat(dataArr_test)
        labelMat_test = np.mat(labelArr_test).T
        m,n = np.shape(datMat_test)
        for j in range(m): #在测试数据上检验错误率
            kernelEval = kernelTrans(sVs,datMat_test[j,:],('rbf', 100))
            predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
            if np.sign(predict)!= np.sign(labelArr_test[j]):
                errorCount_test += 1
        print("the test error rate is: %f" % (float(errorCount_test)/m))


#主程序
def main():
    filename_data='/Users/fmtaxz/PycharmProjects/untitled2/traindata.csv'
    testRbf(filename_data)

if __name__=='__main__':
    main()