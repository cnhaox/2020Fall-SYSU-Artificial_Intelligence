import numpy as np
import random
def calculate_accuracy(X,Y,W):
    """
    计算当前权重W的准确率
    """
    # 计算预测值
    result=np.sign(np.dot(W.T,X))
    # 统计结果
    correct=0
    for i in range(np.size(Y)):
        if result[0,i]==Y[0,i]:
            correct+=1
    # 返回准确率
    return correct/np.size(Y)

def PLA(X, Y, W, Eta, iteration_num):
    """
    PLA算法，计算权重矩阵W
    """
    startIndex=0        # 起始索引
    tempIndex=startIndex# 当前索引
    iterNO=0            # 当前迭代次数
    while iterNO<iteration_num:
        if np.sign(np.dot(W.T,X[:,tempIndex:tempIndex+1]))==Y[:,tempIndex:tempIndex+1]:
        # 如果第tempIndex个样本预测值与实际值一致
            tempIndex=(tempIndex+1)%np.size(Y)# 索引喜加一
            if tempIndex==startIndex:         # 如果所有样本的预测值和实际值均一致
                break                         # 结束迭代
        else:
        # 如果第tempIndex个样本预测值与实际值不一致
            W+=X[:,tempIndex:tempIndex+1]*(Eta*Y[0,tempIndex])#更新权重矩阵
            iterNO+=1                                         #迭代次数加一
            startIndex=tempIndex
            #if iterNO%20==0:
                #print('#'+str(iterNO)+': done')
    #返回迭代后的权重矩阵
    return W

def PLA_temp(X, Y, W, Eta, iteration_num, interval):
    """
    PLA算法，计算不同参数下的权重大矩阵W
    """
    Wnum=int(iteration_num/interval)
    if iteration_num%interval>0:
        Wnum+=1
    totalW=np.zeros((np.size(W),Wnum))
    startIndex=0        # 起始索引
    tempIndex=startIndex# 当前索引
    iterNO=0            # 当前迭代次数
    while iterNO<iteration_num:
        if np.sign(np.dot(W.T,X[:,tempIndex:tempIndex+1]))==Y[:,tempIndex:tempIndex+1]:
        # 如果第tempIndex个样本预测值与实际值一致
            tempIndex=(tempIndex+1)%np.size(Y)# 索引喜加一
            if tempIndex==startIndex:         # 如果所有样本的预测值和实际值均一致
                break                         # 结束迭代
        else:
        # 如果第tempIndex个样本预测值与实际值不一致
            W+=X[:,tempIndex:tempIndex+1]*(Eta*Y[0,tempIndex])#更新权重矩阵
            iterNO+=1                                         #迭代次数加一
            startIndex=tempIndex
            if (iterNO)%interval==0:
                print(str(Eta)+':'+str(iterNO)+': done')
                for j in range(np.size(W)):
                    totalW[j,int((iterNO)/interval)-1]=W[j,0]
    #返回迭代后的权重矩阵
    return totalW

def predict(W,X):
    """
    预测数据集X的值
    """
    # 计算预测值
    return np.sign(np.dot(W.T,X))

# 读取数据
data=np.loadtxt('AIlab3/train.csv', delimiter=',')
features=data[:,:-1]
Y=data[:,-1:]
for i in range(np.size(Y)):
    if Y[i,0]!=1:
        Y[i,0]=-1
biasX=np.ones((np.size(Y),1))
X=np.c_[biasX, features]
# 划分训练集和验证集
trainNum=int(np.size(Y)*0.7)
trainX=(X[:trainNum,:]).T
trainY=(Y[:trainNum,:]).T
ValidX=(X[trainNum:,:]).T
ValidY=(Y[trainNum:,:]).T
# 设置相关参数
random.seed(0)
np.random.seed(0)
#iteration_num=15000# 最大迭代次数
#interval=25        # 间隔
#accuracy=np.zeros((8,int(iteration_num/interval)))
#W=np.zeros((np.size(trainX,axis=0),1))# 初始权重矩阵
#tW=np.random.rand(np.size(trainX,axis=0))
#tW=np.reshape(tW,(np.size(trainX,axis=0),1))
#for a in range(8):
#    W=tW.copy()
#    totalW=PLA_temp(trainX, trainY, W, (a+1)*0.25, iteration_num,interval)
#    for b in range(int(iteration_num/interval)):
#        accuracy[a,b]=calculate_accuracy(ValidX,ValidY,totalW[:,b:b+1])
#    print('('+str(a)+', '+')'+': finished')
#np.savetxt('AIlab3/1accuracy3.txt',accuracy,fmt='%0.8f')
W=np.zeros((np.size(trainX,axis=0),1))# 初始权重矩阵
W=PLA(trainX, trainY, W, 1, 1350)# 训练W
print(calculate_accuracy(ValidX,ValidY,W))# 对验证集准确率
prediction=predict(W, ValidX)# 得到预测值
print(prediction)