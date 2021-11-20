import numpy as np
import math
import random
def calculate_accuracy(X,Y,W, threshold):
    """
    计算当前权重W的准确率
    """
    # 计算预测值
    regression=calculate_p(W,X)
    # 统计结果
    correct=0
    total=np.size(Y)
    for i in range(total):
        # 根据阈值预测类别
        if regression[0,i]>threshold:
            regression[0,i]=1
        else:
            regression[0,i]=0
        # 判断是否预测正确
        if Y[0,i]==regression[0,i]:
            correct+=1
    # 返回准确率
    return correct/total

def predict(W,X,threshold):
    """
    计算预测值
    """
    # 计算预测值
    regression=calculate_p(W,X)
    total=np.size(regression)
    for i in range(total):
        # 根据阈值预测类别
        if regression[0,i]>threshold:
            regression[0,i]=1
        else:
            regression[0,i]=0
    return regression

def calculate_loss(X, Y, W):
    """
    计算损失函数
    """
    temp1=np.dot(W.T,X)
    temp2=np.log((np.exp(temp1)+1))
    return -np.sum(temp1*Y-temp2)

def calculate_p(W,X):
    """
    计算π(x)，即逻辑回归预测为1的概率
    """
    temp=np.dot(W.T,X)
    p=np.zeros_like(temp)
    for i in range(np.size(temp)):
        # 根据e^(wx+b)的正负，采用不同方式计算p，避免出现数值溢出
        if temp[0,i]>0:
            p[0,i]=1/(1+math.exp(-temp[0,i]))
        else:
            p[0,i]=math.exp(temp[0,i])/(math.exp(temp[0,i])+1)
    return p

def Logistics_regression(X,Y,W,Eta,iteration_num):
    """
    逻辑回归，计算权重矩阵W
    """
    #result=np.zeros((iteration_num,1))
    for i in range(iteration_num):
        p=calculate_p(W,X)  # 计算π(x)
        temp=Y-p
        temp2=temp*X
        temp3=np.reshape(np.sum(temp2,axis=1),(np.size(W),1))
        W+=temp3*Eta        # 更新参数
        #print(result[i,0])
    # 返回更新后的参数
    return W

def Logistics_regression_temp(X,Y,W,Eta,iteration_num, interval):
    """
    逻辑回归，计算不同参数下的权重大矩阵W
    """
    Wnum=int(iteration_num/interval)
    if iteration_num%interval>0:
        Wnum+=1
    totalW=np.zeros((np.size(W),Wnum))
    for i in range(iteration_num):
        p=calculate_p(W,X)  # 计算π(x)
        temp=Y-p
        temp2=temp*X
        temp3=np.reshape(np.sum(temp2,axis=1),(np.size(W),1))
        W+=temp3*Eta        # 更新参数
        if (i+1)%20==0:
            print(str(Eta)+':'+str(i+1)+': done')
        if (i+1)%interval==0:
            for j in range(np.size(W)):
                totalW[j,int((i+1)/interval)-1]=W[j,0]
    # 返回更新后的参数
    return totalW

# 读取数据
data=np.loadtxt('AIlab3/train.csv', delimiter=',')
features=data[:,:-1]
Y=data[:,-1:]
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
#iteration_num=100 # 最大迭代次数
#interval=1        # 间隔
#accuracy=np.zeros((1,int(iteration_num/interval)))
#tW=np.random.rand(np.size(trainX,axis=0))# 初始权重矩阵
#tW=np.reshape(tW,(np.size(trainX,axis=0),1))
#for a in range(1):
#    #W=tW.copy()
#    W=np.ones((np.size(trainX,axis=0),1))
#    totalW=Logistics_regression_temp(trainX, trainY, W, (a+1)*0.005, iteration_num,interval)
#    for b in range(int(iteration_num/interval)):
#        accuracy[a,b]=calculate_accuracy(ValidX,ValidY,totalW[:,b:b+1],0.5)
#    print('('+str(a)+', '+')'+': finished')
#np.savetxt('AIlab3/2accuracy6.txt',accuracy,fmt='%0.8f')
W=np.zeros((np.size(trainX,axis=0),1))# 初始权重矩阵
W=Logistics_regression(trainX, trainY, W, 1, 70)# 训练W
print(calculate_accuracy(ValidX,ValidY,W,0.5))# 对验证集准确率
prediction=predict(W, ValidX, 0.5)# 得到预测值
print(prediction)