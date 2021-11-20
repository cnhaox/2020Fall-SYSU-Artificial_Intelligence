import math
import copy
def build_WordDict(FileText):
    """
    构建TF-IDF字典 {word:编码列表}
    """
    WordDict=dict()                 #字典
    TextSize=len(FileText)          #文档数量
    TextIndex=0
    WordNumPerSen=[0]*TextSize      #文档单词数量列表
    #构建词频字典
    for text in FileText:
        words=text.split()   #将句子分割成单词列表
        WordNumPerSen[TextIndex]+=len(words)
        for word in words:
            if word not in WordDict:#如果单词不在字典中
                WordDict[word]=[0]*TextSize
            WordDict[word][TextIndex]+=1
        TextIndex+=1
    #构建TF-IDF字典
    for word in WordDict:
        C_i=1
        #统计出现word的文档数
        for i in range(len(WordDict[word])):
            if WordDict[word][i]>0:
                C_i+=1
        #计算word的idf
        idf_i=math.log(TextSize/C_i)
        for i in range(len(WordDict[word])):
            #计算word的tf
            WordDict[word][i]/=WordNumPerSen[i]
            #计算word的tf-idf
            WordDict[word][i]*=idf_i
    return WordDict

def build_EmotionDict(FileText):
    """
    构建emotion字典 {emotion:概率列表}
    """
    EmotionDict=dict()          #emotion字典
    TextSize=len(FileText)
    TextIndex=0
    for text in FileText:
        TextList=text.split(',')
        for index in range(len(TextList)):
            if index not in EmotionDict:
                EmotionDict[index]=[0]*TextSize
            EmotionDict[index][TextIndex]=float(TextList[index])
        TextIndex+=1
    return EmotionDict

def calculate_distance_p(WordDist, index1, index2, p):
    """
    计算LP距离
    """
    distance=0
    for word in WordDist.keys():
        distance+=pow(abs(WordDist[word][index1] - WordDist[word][index2]),p)
    distance=pow(distance,1/p)
    return distance

def calculate_distance_cos(WordDist, index1, index2):
    """
    计算余弦距离
    """
    AB=0
    AA=0
    BB=0
    for word in WordDist.keys():
        AA+=pow(WordDist[word][index1], 2)
        AB+=WordDist[word][index1]*WordDist[word][index2]
        BB+=pow(WordDist[word][index2],2)
    AA=pow(AA,0.5)
    BB=pow(BB,0.5)
    distance=1-AB/(AA*BB)
    return distance 

def predict_validation(WordDict,EmotionDict,TrainStart,TrainSize,ValidStart,ValidSize,DistanceType,K):
    """
    对验证集预测验证
    """
    PredEmotionDict=dict()
    for emotion in EmotionDict.keys():
        PredEmotionDict[emotion]=[]
    #对每个验证集句子预测
    for i in range(ValidStart, ValidStart+ValidSize):
        distance=dict()         #该验证集句子与训练集各句的距离
        #计算该验证集句子与训练集各句的距离
        for j in range(TrainStart, TrainStart+TrainSize):
            if DistanceType==0:
                distance[j]=calculate_distance_cos(WordDict, j, i)
            else:
                distance[j]=calculate_distance_p(WordDict, j, i, DistanceType)
        #对距离进行排序
        distance_order=sorted(distance.items(),key=lambda x:x[1],reverse=False)
        #对各种情感进行预测
        for emotion in PredEmotionDict.keys():
            PredEmotionDict[emotion].append(0)
            for j in range(K):
                if distance_order[j][1]==0:#如果距离为0，直接学习
                    PredEmotionDict[emotion][-1]=EmotionDict[emotion][distance_order[j][0]]
                    break
                else:#如果距离不为0，计算情感概率
                    PredEmotionDict[emotion][-1]+=EmotionDict[emotion][distance_order[j][0]]/distance_order[j][1]
        #修改情感概率，使它们和为0
        EmotionSum=0
        for emotion in EmotionDict.keys():
            EmotionSum+=PredEmotionDict[emotion][-1]
        for emotion in EmotionDict.keys():
            PredEmotionDict[emotion][-1]/=EmotionSum
        if (i-TrainSize)%20==0:
            print(str(DistanceType)+'|'+str(K)+':'+str(i-TrainSize)+': done')
    #计算真实情感概率和预测情感概率的相关系数
    COR=0
    for emotion in EmotionDict.keys():
        avg_TrueVal=sum(EmotionDict[emotion][ValidStart:ValidStart+ValidSize])/ValidSize
        avg_PredVal=sum(PredEmotionDict[emotion])/ValidSize
        cov=0
        sigmaX=0
        sigmaY=0
        for i in range(ValidSize):
            X=EmotionDict[emotion][i+ValidStart]-avg_TrueVal
            Y=PredEmotionDict[emotion][i]-avg_PredVal
            cov+=X*Y
            sigmaX+=pow(X,2)
            sigmaY+=pow(Y,2)
        COR+=cov/pow(sigmaX*sigmaY,0.5)
    #计算最终相关系数
    COR/=len(EmotionDict.keys())
    return COR#返回相关系数

def predict_test(WordDict,EmotionDict,TrainStart,TrainSize,TestStart,TestSize,DistanceType,K):
    """
    对测试集预测
    """
    PredEmotionDict=dict()
    for emotion in EmotionDict.keys():
        PredEmotionDict[emotion]=[]
    #对每个测试集句子预测
    for i in range(TestStart, TestStart+TestSize):
        distance=dict()         #该测试集句子与训练集各句的距离
        #计算该测试集句子与训练集各句的距离
        for j in range(TrainStart, TrainStart+TrainSize):
            if DistanceType==0:
                distance[j]=calculate_distance_cos(WordDict, j, i)
            else:
                distance[j]=calculate_distance_p(WordDict, j, i, DistanceType)
        #对距离进行排序
        distance_order=sorted(distance.items(),key=lambda x:x[1],reverse=False)
        #对各种情感进行预测
        for emotion in PredEmotionDict.keys():
            PredEmotionDict[emotion].append(0)
            for j in range(K):
                if distance_order[j][1]==0:#如果距离为0，直接学习
                    PredEmotionDict[emotion][-1]=EmotionDict[emotion][distance_order[j][0]]
                    break
                else:#如果距离不为0，计算情感概率
                    PredEmotionDict[emotion][-1]+=EmotionDict[emotion][distance_order[j][0]]/distance_order[j][1]
        #修改情感概率，使它们和为0
        EmotionSum=0
        for emotion in EmotionDict.keys():
            EmotionSum+=PredEmotionDict[emotion][-1]
        for emotion in EmotionDict.keys():
            PredEmotionDict[emotion][-1]/=EmotionSum
        if (i-TestStart)%20==0:
            print(str(DistanceType)+'|'+str(K)+':'+str(i-TestStart)+': done')
    return PredEmotionDict

    
with open("regression_dataset/train_set.csv") as tf:
    TrainFileText=tf.readlines()
with open("regression_dataset/validation_set.csv") as vf:
    ValidFileText=vf.readlines()
with open("regression_dataset/test_set.csv") as tf:
    TestFileText=tf.readlines()
SentenceList=[]
EmotionList=[]

TrainStart=0
TrainSize=len(TrainFileText)-1
for text in TrainFileText[1:]:
    textlist=text.split(',')
    SentenceList.append(textlist[0])
    EmotionList.append(text[len(textlist[0])+1:])

ValidStart=TrainSize
ValidSize=len(ValidFileText)-1
for text in ValidFileText[1:]:
    textlist=text.split(',')
    SentenceList.append(textlist[0])
    EmotionList.append(text[len(textlist[0])+1:])

TestStart=ValidSize+TrainSize
TestSize=len(TestFileText)-1
for text in TestFileText[1:]:
    textlist=text.split(',')
    SentenceList.append(textlist[1])

WordDict=build_WordDict(SentenceList)
EmotionDict=build_EmotionDict(EmotionList)

#accuracy=[]
#for DistanceType in range(4):
#    accuracy.append([])
#    for K in range(25,27,2):
#        accuracy[DistanceType].append(predict_validation(WordDict, EmotionDict, TrainStart, TrainSize, ValidStart, ValidSize, DistanceType, K))
#
#with open('KNNregression_prediction4.txt','w') as f:
#    for j in range(len(accuracy[0])):
#        for i in range(len(accuracy)):
#            f.write(str(round(accuracy[i][j],5))+' ')
#        f.write('\n')

TestEmotion=predict_test(WordDict,EmotionDict,TrainStart,TrainSize,TestStart,TestSize,0,7)
with open("18308013_ChenJiahao_KNN_regression.csv",'a+') as f:
    f.write('textid,Words (split by space),anger,disgust,fear,joy,sad,surprise\n')
    for i in range(len(TestEmotion[0])):
        f.write(str(i+1)+','+SentenceList[i+TestStart]+','+str(round(TestEmotion[0][i],4))+','+str(round(TestEmotion[1][i],4))+','+str(round(TestEmotion[2][i],4))+','+str(round(TestEmotion[3][i],4))+','+str(round(TestEmotion[4][i],4))+','+str(round(TestEmotion[5][i],4))+'\n')