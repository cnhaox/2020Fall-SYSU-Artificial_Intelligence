import math
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
    构建emotion字典 {文档序号:emotion}
    """
    EmotionDict=dict()          #emotion字典
    TextIndex=0
    for text in FileText:
        EmotionDict[TextIndex]=text
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

def predict_validation(WordDict, EmotionDict, TrainStart, TrainSize, ValidStart, ValidSize, DistanceType, K):
    """
    对验证集预测验证
    """
    correct=0       #预测正确个数
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
        prediction=dict()   #投票字典
        #选取前K个最近的距离
        for j in range(K):
            #进行投票
            if EmotionDict[distance_order[j][0]] not in prediction.keys():
                prediction[EmotionDict[distance_order[j][0]]]=1
            else:
                prediction[EmotionDict[distance_order[j][0]]]+=1
        #对投票结果进行排序
        prediction_order=sorted(prediction.items(),key=lambda x:x[1],reverse=True)
        if prediction_order[0][0]==EmotionDict[i]:#如果预测值与实际值相同
            correct+=1
        if (i-TrainSize)%20==0:
            print(str(DistanceType)+'|'+str(K)+':'+str(i-TrainSize)+': done')
    return correct/ValidSize#返回准确率

def predict_test(WordDict, EmotionDict, TrainStart, TrainSize, TestStart, TestSize, DistanceType, K):
    """
    对测试集预测
    """
    TestEmotion=[]  #预测结果
    #对每个测试集句子预测
    for i in range(TestStart, TestStart+TestSize):
        distance=dict()
        #计算该测试集句子与训练集各句的距离
        for j in range(TrainStart, TrainStart+TrainSize):
            if DistanceType==0:
                distance[j]=calculate_distance_cos(WordDict, j, i)
            else:
                distance[j]=calculate_distance_p(WordDict, j, i, DistanceType)
        #对距离进行排序
        distance_order=sorted(distance.items(),key=lambda x:x[1],reverse=False)
        prediction=dict()   #投票字典
        #选取前K个最近的距离
        for j in range(K):
            #进行投票
            if EmotionDict[distance_order[j][0]] not in prediction.keys():
                prediction[EmotionDict[distance_order[j][0]]]=1
            else:
                prediction[EmotionDict[distance_order[j][0]]]+=1
        #对投票结果进行排序
        prediction_order=sorted(prediction.items(),key=lambda x:x[1],reverse=True)
        TestEmotion.append(prediction_order[0][0])  #将预测结果加入列表
        if (i-TestStart)%20==0:
            print(str(DistanceType)+'|'+str(K)+':'+str(i-TestStart)+': done')
    return TestEmotion#返回预测列表


with open("classification_dataset/train_set.csv") as tf:
    TrainFileText=tf.readlines()
with open("classification_dataset/validation_set.csv") as vf:
    ValidFileText=vf.readlines()
with open("classification_dataset/test_set.csv") as tf:
    TestFileText=tf.readlines()
SentenceList=[]
EmotionList=[]

TrainStart=0
TrainSize=len(TrainFileText)-1
for text in TrainFileText[1:]:
    textlist=text.split(',')
    SentenceList.append(textlist[0])
    EmotionList.append(textlist[1])

ValidStart=TrainSize
ValidSize=len(ValidFileText)-1
for text in ValidFileText[1:]:
    textlist=text.split(',')
    SentenceList.append(textlist[0])
    EmotionList.append(textlist[1])

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
#with open('KNNclassification_prediction3.txt','w') as f:
#    for j in range(len(accuracy[0])):
#        for i in range(len(accuracy)):
#            f.write(str(round(accuracy[i][j],5))+' ')
#        f.write('\n')

TestEmotion=predict_test(WordDict,EmotionDict,TrainStart,TrainSize,TestStart,TestSize,0,7)
with open("18308013_ChenJiahao_KNN_classification.csv",'a+') as f:
    f.write('textid,Words (split by space),label\n')
    for i in range(len(TestEmotion)):
        f.write(str(i+1)+','+SentenceList[i+TestStart]+','+TestEmotion[i])