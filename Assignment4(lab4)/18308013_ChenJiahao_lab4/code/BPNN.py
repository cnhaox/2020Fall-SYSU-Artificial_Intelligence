import math
import numpy as np

def sigmoid(x):
    """
    sigmoid激活函数
    """
    if x>0:
        return 1/(1+np.exp(-x))
    else:
        return np.exp(x)/(1+np.exp(x))

def tanh(x):
    """
    tanh激活函数
    """
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def ReLU(x,a):
    """
    ReLU函数
    """
    return max(a*x,x)

def dReLU(matrix,a):
    """
    对ReLU函数求导
    """
    dmatrix=np.zeros_like(matrix)
    for index1 in range(np.size(matrix,axis=0)):
        for index2 in range(np.size(matrix,axis=1)):
            if matrix[index1,index2]>0:
                dmatrix[index1,index2]=1
            else:
                dmatrix[index1,index2]=a
    return dmatrix

def train_network(X, Y, input_nodes, hidden_nodes, output_nodes, learning_rate, iterations):
    """
    训练三层网络
    """
    np.random.seed(0)
    # 初始化权重矩阵
    W_input_hidden=np.random.normal(loc=0,scale=1,size=(input_nodes+1,hidden_nodes))
    W_hidden_output=np.random.normal(loc=0,scale=1,size=(hidden_nodes+1,output_nodes))
    m=np.size(X,axis=0)

    for iter in range(iterations):
        # 前向传播
        # 输入层
        input_X=X
        input_H=input_X
        input_F=input_H
        # 隐藏层
        addMatrix=np.ones((np.size(input_F,axis=0), 1))
        hidden_X=np.column_stack((input_F,addMatrix))
        hidden_H=np.dot(hidden_X, W_input_hidden)
        hidden_F=np.zeros(hidden_H.shape)
        for i in range(np.size(hidden_F,axis=0)):
            for j in range(np.size(hidden_F,axis=1)):
                hidden_F[i,j]=sigmoid(hidden_H[i,j])
        # 输出层
        addMatrix=np.ones((np.size(hidden_F,axis=0), 1))
        output_X=np.column_stack((hidden_F,addMatrix))
        output_H=np.dot(output_X, W_hidden_output)
        output_Y=output_H

        # 反向传播
        error=output_Y-Y
        output_error=error
        # 输出层
        dW_hidden_output=np.dot(output_X.T, output_error)
        hidden_error=np.dot(output_error, W_hidden_output.T)
        hidden_error=hidden_error[:,:-1]
        # 隐藏层
        dW_input_hidden=np.dot(hidden_X.T, hidden_F*(1-hidden_F)*hidden_error)
        # 更新权重
        W_input_hidden-=dW_input_hidden*learning_rate/m
        W_hidden_output-=dW_hidden_output*learning_rate/m

        if iter%10==0:
            print('#'+str(iter)+' done')
    # 返回训练好的权重矩阵
    return W_input_hidden, W_hidden_output

def train_network2(X, Y, input_nodes, hidden_nodes1, hidden_nodes2, output_nodes, learning_rate, iterations):
    """
    训练四层网络
    """
    np.random.seed(0)
    # 初始化权重矩阵
    W_input_hidden1=np.random.normal(loc=0,scale=1,size=(input_nodes+1,hidden_nodes1))
    W_hidden1_hidden2=np.random.normal(loc=0,scale=1,size=(hidden_nodes1+1,hidden_nodes2))
    W_hidden2_output=np.random.normal(loc=0,scale=1,size=(hidden_nodes2+1,output_nodes))
    m=np.size(X,axis=0)

    for iter in range(iterations):
        # 前向传播
        # 输入层
        input_X=X
        input_H=input_X
        input_F=input_H
        # 隐藏层1
        addMatrix=np.ones((np.size(input_F,axis=0), 1))
        hidden1_X=np.column_stack((input_F,addMatrix))
        hidden1_H=np.dot(hidden1_X, W_input_hidden1)
        hidden1_F=np.zeros(hidden1_H.shape)
        for i in range(np.size(hidden1_F,axis=0)):
            for j in range(np.size(hidden1_F,axis=1)):
                hidden1_F[i,j]=sigmoid(hidden1_H[i,j])
        # 隐藏层2
        addMatrix=np.ones((np.size(hidden1_F,axis=0), 1))
        hidden2_X=np.column_stack((hidden1_F,addMatrix))
        hidden2_H=np.dot(hidden2_X, W_hidden1_hidden2)
        hidden2_F=np.zeros(hidden2_H.shape)
        for i in range(np.size(hidden2_F,axis=0)):
            for j in range(np.size(hidden2_F,axis=1)):
                hidden2_F[i,j]=sigmoid(hidden2_H[i,j])
        # 输出层
        addMatrix=np.ones((np.size(hidden2_F,axis=0), 1))
        output_X=np.column_stack((hidden2_F,addMatrix))
        output_H=np.dot(output_X, W_hidden2_output)
        output_Y=output_H

        # 反向传播
        error=output_Y-Y
        output_error=error
        # 输出层
        dW_hidden2_output=np.dot(output_X.T, output_error)
        hidden2_error=np.dot(output_error, W_hidden2_output.T)
        hidden2_error=hidden2_error[:,:-1]
        # 隐藏层2
        dW_hidden1_hidden2=np.dot(hidden2_X.T, hidden2_F*(1-hidden2_F)*hidden2_error)
        hidden1_error=np.dot(hidden2_error, W_hidden1_hidden2.T)
        hidden1_error=hidden1_error[:,:-1]
        # 隐藏层1
        dW_input_hidden1=np.dot(hidden1_X.T, hidden1_F*(1-hidden1_F)*hidden1_error)
        # 更新权重
        W_input_hidden1-=dW_input_hidden1*learning_rate/m
        W_hidden1_hidden2-=dW_hidden1_hidden2*learning_rate/m
        W_hidden2_output-=dW_hidden2_output*learning_rate/m
        if iter%10==0:
            print('#'+str(iter)+' done')
    # 返回训练好的权重矩阵
    return W_input_hidden1, W_hidden1_hidden2, W_hidden2_output

def train_network3(X, Y, input_nodes, hidden_nodes1, hidden_nodes2, hidden_nodes3, output_nodes, learning_rate, iterations):
    """
    训练网络代码
    """
    np.random.seed(0)
    # 初始化权重矩阵
    W_input_hidden1=np.random.normal(loc=0,scale=1,size=(input_nodes+1,hidden_nodes1))
    W_hidden1_hidden2=np.random.normal(loc=0,scale=1,size=(hidden_nodes1+1,hidden_nodes2))
    W_hidden2_hidden3=np.random.normal(loc=0,scale=1,size=(hidden_nodes2+1,hidden_nodes3))
    W_hidden3_output=np.random.normal(loc=0,scale=1,size=(hidden_nodes3+1,output_nodes))
    m=np.size(X,axis=0)

    for iter in range(iterations):
        # 前向传播
        # 输入层
        input_X=X
        input_H=input_X
        input_F=input_H
        # 隐藏层1
        addMatrix=np.ones((np.size(input_F,axis=0), 1))
        hidden1_X=np.column_stack((input_F,addMatrix))
        hidden1_H=np.dot(hidden1_X, W_input_hidden1)
        hidden1_F=np.zeros(hidden1_H.shape)
        for i in range(np.size(hidden1_F,axis=0)):
            for j in range(np.size(hidden1_F,axis=1)):
                hidden1_F[i,j]=sigmoid(hidden1_H[i,j])
        # 隐藏层2
        addMatrix=np.ones((np.size(hidden1_F,axis=0), 1))
        hidden2_X=np.column_stack((hidden1_F,addMatrix))
        hidden2_H=np.dot(hidden2_X, W_hidden1_hidden2)
        hidden2_F=np.zeros(hidden2_H.shape)
        for i in range(np.size(hidden2_F,axis=0)):
            for j in range(np.size(hidden2_F,axis=1)):
                hidden2_F[i,j]=sigmoid(hidden2_H[i,j])
        # 隐藏层3
        addMatrix=np.ones((np.size(hidden2_F,axis=0), 1))
        hidden3_X=np.column_stack((hidden2_F,addMatrix))
        hidden3_H=np.dot(hidden3_X, W_hidden2_hidden3)
        hidden3_F=np.zeros(hidden3_H.shape)
        for i in range(np.size(hidden3_F,axis=0)):
            for j in range(np.size(hidden3_F,axis=1)):
                hidden3_F[i,j]=sigmoid(hidden3_H[i,j])
        # 输出层
        addMatrix=np.ones((np.size(hidden3_F,axis=0), 1))
        output_X=np.column_stack((hidden3_F,addMatrix))
        output_H=np.dot(output_X, W_hidden3_output)
        output_Y=output_H

        # 反向传播
        error=output_Y-Y
        output_error=error
        # 输出层
        dW_hidden3_output=np.dot(output_X.T, output_error)
        hidden3_error=np.dot(output_error, W_hidden3_output.T)
        hidden3_error=hidden3_error[:,:-1]
        # 隐藏层3
        dW_hidden2_hidden3=np.dot(hidden3_X.T, hidden3_F*(1-hidden3_F)*hidden3_error)
        hidden2_error=np.dot(hidden3_error, W_hidden2_hidden3.T)
        hidden2_error=hidden2_error[:,:-1]
        # 隐藏层2
        dW_hidden1_hidden2=np.dot(hidden2_X.T, hidden2_F*(1-hidden2_F)*hidden2_error)
        hidden1_error=np.dot(hidden2_error, W_hidden1_hidden2.T)
        hidden1_error=hidden1_error[:,:-1]
        # 隐藏层1
        dW_input_hidden1=np.dot(hidden1_X.T, hidden1_F*(1-hidden1_F)*hidden1_error)
        # 更新权重
        W_input_hidden1-=dW_input_hidden1*learning_rate/m
        W_hidden1_hidden2-=dW_hidden1_hidden2*learning_rate/m
        W_hidden2_hidden3-=dW_hidden2_hidden3*learning_rate/m
        W_hidden3_output-=dW_hidden3_output*learning_rate/m
        if iter%10==0:
            print('#'+str(iter)+' done')
    # 返回训练好的权重矩阵
    return W_input_hidden1, W_hidden1_hidden2, W_hidden2_hidden3, W_hidden3_output

def predict(X, W_input_hidden, W_hidden_output):
    """
    三层网络的预测
    """
    # 输入层
    X1=X
    H1=X1
    F1=H1
    # 隐藏层
    addMatrix=np.ones((np.size(F1,axis=0), 1))
    X2=np.column_stack((F1,addMatrix))
    H2=np.dot(X2, W_input_hidden)
    F2=np.zeros(H2.shape)
    for i in range(np.size(F2,axis=0)):
        for j in range(np.size(F2,axis=1)):
            F2[i,j]=sigmoid(H2[i,j])
    # 输出层
    addMatrix=np.ones((np.size(F2,axis=0), 1))
    X3=np.column_stack((F2,addMatrix))
    H3=np.dot(X3, W_hidden_output)
    Y3=H3
    return Y3

def predict2(X, W_input_hidden1, W_hidden1_hidden2, W_hidden2_output):
    """
    四层网络的预测
    """
    # 输入层
    X0=X
    H0=X0
    F0=H0
    # 隐藏层1
    addMatrix=np.ones((np.size(F0,axis=0), 1))
    X1=np.column_stack((F0,addMatrix))
    H1=np.dot(X1, W_input_hidden1)
    F1=np.zeros(H1.shape)
    for i in range(np.size(F1,axis=0)):
        for j in range(np.size(F1,axis=1)):
            F1[i,j]=sigmoid(H1[i,j])
    # 隐藏层2
    addMatrix=np.ones((np.size(F1,axis=0), 1))
    X2=np.column_stack((F1,addMatrix))
    H2=np.dot(X2, W_hidden1_hidden2)
    F2=np.zeros(H2.shape)
    for i in range(np.size(F2,axis=0)):
        for j in range(np.size(F2,axis=1)):
            F2[i,j]=sigmoid(H2[i,j])
    # 输出层
    addMatrix=np.ones((np.size(F2,axis=0), 1))
    X3=np.column_stack((F2,addMatrix))
    H3=np.dot(X3, W_hidden2_output)
    Y3=H3
    return Y3

def predict3(X, W_input_hidden1, W_hidden1_hidden2, W_hidden2_hidden3, W_hidden3_output):
    """
    五层网络的预测
    """
    # 输入层
    X0=X
    H0=X0
    F0=H0
    # 隐藏层1
    addMatrix=np.ones((np.size(F0,axis=0), 1))
    X1=np.column_stack((F0,addMatrix))
    H1=np.dot(X1, W_input_hidden1)
    F1=np.zeros(H1.shape)
    for i in range(np.size(F1,axis=0)):
        for j in range(np.size(F1,axis=1)):
            F1[i,j]=sigmoid(H1[i,j])
    # 隐藏层2
    addMatrix=np.ones((np.size(F1,axis=0), 1))
    X2=np.column_stack((F1,addMatrix))
    H2=np.dot(X2, W_hidden1_hidden2)
    F2=np.zeros(H2.shape)
    for i in range(np.size(F2,axis=0)):
        for j in range(np.size(F2,axis=1)):
            F2[i,j]=sigmoid(H2[i,j])
    # 隐藏层3
    addMatrix=np.ones((np.size(F2,axis=0), 1))
    X3=np.column_stack((F2,addMatrix))
    H3=np.dot(X3, W_hidden2_hidden3)
    F3=np.zeros(H3.shape)
    for i in range(np.size(F3,axis=0)):
        for j in range(np.size(F3,axis=1)):
            F3[i,j]=sigmoid(H3[i,j])
    # 输出层
    addMatrix=np.ones((np.size(F3,axis=0), 1))
    X4=np.column_stack((F3,addMatrix))
    H4=np.dot(X4, W_hidden3_output)
    Y4=H4
    return Y4


# 读取数据
with open('AIlab4/lab4_dataset/train.csv') as tf:
    FileText=tf.readlines()         # 读取文本
samples_num=len(FileText)-1         # 总样本数量
# 数据预处理
# season:4 yr:1 mnth:12 hr:24 holiday:1 weekday:7 workingday:1
# weathersit:4 temp:1 atemp:1 hum:1 windspeed:1
X=np.zeros((58,samples_num))        # 特征矩阵：特征数量*样本数量
Y=np.zeros((1,samples_num))         # 结果矩阵：1*样本数量
for i in range(samples_num):        # 对每行样本文本处理
    temp=FileText[i+1].split(',')
    X[int(temp[2])-1,i]=1           # season转换成one-hot编码
    X[4,i]=float(temp[3])           # yr不做转换
    X[int(temp[4])+4,i]=1           # mnth转换成one-hot编码
    X[int(temp[5])+17,i]=1          # hr转换成one-hot编码
    X[41,i]=float(temp[6])          # holiday不做转换
    X[int(temp[7])+42,i]=1          # weekday转换成one-hot编码
    X[49,i]=float(temp[8])          # workingday不做转换
    X[int(temp[9])+49,i]=1          # weathersit转换成one-hot编码
    X[54,i]=float(temp[10])         # temp不做转换
    X[55,i]=float(temp[11])         # atemp不做转换
    X[56,i]=float(temp[12])         # hum不做转换
    X[57,i]=float(temp[13])         # windspeed不做转换
    Y[0,i]=float(temp[14])          # 真实值直接读取
X=X.T                               # 特征矩阵：样本数量*特征数量
Y=Y.T                               # 结果数量：样本数量*1

# 对几个数值大的特征和结果标准化、进行缩放
X_mean=np.mean(X,axis=0)            # 特征平均值
X_std=np.std(X,axis=0)              # 特征方差
for i in range(54):                 # 有些特征不需要标准化
    X_mean[i]=0
    X_std[i]=1
Y_mean=np.mean(Y,axis=0)            # 结果平均值
Y_std=np.std(Y,axis=0)              # 结果方差
X=(X-X_mean)/X_std                  # 特征标准化
Y=(Y-Y_mean)/Y_std                  # 结果标准化

# 分割数据集
TrainX=X[:int(samples_num*0.8),:]
TrainY=Y[:int(samples_num*0.8),:]
ValidX=X[int(samples_num*0.8):,:]
ValidY=Y[int(samples_num*0.8):,:]
# 训练三层网络
W_input_hidden, W_hidden_output=train_network(TrainX, TrainY, 58, 49, 1, 0.20, 3600)
# 预测结果
predictY=predict(X,W_input_hidden,W_hidden_output)
# 还原单车预测值
predictY=predictY*Y_std+Y_mean
# 存储
np.savetxt('predict.txt',predictY,fmt='%0.1f')