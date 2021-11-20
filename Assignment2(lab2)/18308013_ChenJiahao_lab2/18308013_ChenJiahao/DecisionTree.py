import math
import copy
def Calculate_Entropy(DataLists,TypeUsedList,TypeValList):
    """
    计算信息熵
    """
    pos_num=0                   #当前数据集中样本标签为1的数量
    total_num=0                 #当前数据集样本数量
    for DataList in DataLists:
        #判断当前样本是否位于当前数据集
        isSatisfy=True
        for i in range(len(TypeUsedList)):
            if TypeUsedList[i]==True:
                if DataList[i] in TypeValList[i]:
                    continue
                else:
                    isSatisfy=False
                    break
        if isSatisfy:#如果当前样本位于数据集
            total_num+=1
            if DataList[-1]=='1':
                pos_num+=1
    p_pos=pos_num/total_num
    p_neg=(total_num-pos_num)/total_num
    if p_pos!=0 and p_neg!=0:
        return -p_pos*math.log2(p_pos)-p_neg*math.log2(p_neg)
    else:#如果数据集标签全为1或0，信息熵为0
        return 0

def Calculate_Gain(DataLists,TypeUsedList,TypeValList,FatherTypeIndex):
    """
    计算信息增益
    """
    FatherValDict=dict()        #字典{特征属性各取值:对应样本数量}
    total_num=0                 #当前数据集样本数量
    for DataList in DataLists:
        #判断当前样本是否位于当前数据集
        isSatisfy=True
        for i in range(len(TypeUsedList)):
            if TypeUsedList[i]==True:
                if DataList[i] in TypeValList[i]:
                    continue
                else:#如果该属性不符
                    isSatisfy=False
                    break
        if isSatisfy:#如果当前样本位于数据集
            total_num+=1
            #统计属于暂定特征属性相应取值的样本数量
            if DataList[FatherTypeIndex] not in FatherValDict.keys():
                FatherValDict[DataList[FatherTypeIndex]]=1
            else:
                FatherValDict[DataList[FatherTypeIndex]]+=1
    #计算当前信息熵
    result=Calculate_Entropy(DataLists, TypeUsedList, TypeValList)
    SonTypeUsedList=copy.deepcopy(TypeUsedList)
    SonTypeUsedList[FatherTypeIndex]=True
    for val in FatherValDict.keys():
        SonTypeValList=copy.deepcopy(TypeValList)
        SonTypeValList[FatherTypeIndex].append(val)
        #计算各个子数据集的信息熵
        temp=Calculate_Entropy(DataLists, SonTypeUsedList, SonTypeValList)
        result-=temp*FatherValDict[val]/total_num
    return result

def Calculate_GainRatio(DataLists,TypeUsedList,TypeValList,FatherTypeIndex):
    """
    计算信息增益率
    """
    total_num=0                 #当前数据集样本数量
    TypeValDict=dict()          #字典{特征属性各取值:对应样本数量}
    for datalist in DataLists:
        #判断当前样本是否位于当前数据集
        isSatisfy=True
        for i in range(len(TypeUsedList)):
            if TypeUsedList[i]==True:
                if datalist[i] in TypeValList[i]:
                    continue
                else:
                    isSatisfy=False
                    break
        if isSatisfy:#如果属于当前数据集
            total_num+=1
            #统计属于暂定特征属性相应取值的样本数量
            if datalist[FatherTypeIndex] not in TypeValDict.keys():
                TypeValDict[FatherTypeIndex]=1
            else:
                TypeValDict[FatherTypeIndex]+=1
    #计算得到暂定特征属性的信息增益
    Gain=Calculate_Gain(DataLists,TypeUsedList,TypeValList,FatherTypeIndex)
    #计算数据集关于暂定特征的熵
    SplitInfo=0
    for val in TypeValDict.keys():
        if TypeValDict[val]!=0:
            SplitInfo-=(TypeValDict[val]/total_num)*math.log2(TypeValDict[val]/total_num)
    return Gain/SplitInfo

def Calculate_GINI(DataLists,TypeUsedList,TypeValList,FatherTypeIndex, ChosenVal):
    """
    计算基尼指数
    """
    total_num=0                 #当前数据集样本数量
    NumList=[[0,0],[0,0]]       #划分后两个子数据集的正负样本数量
    for datalist in DataLists:
        isSatisfy=True
        #判断当前样本是否位于当前数据集
        for i in range(len(TypeUsedList)):
            if TypeUsedList[i]==True:
                if datalist[i] in TypeValList[i]:
                    continue
                else:
                    isSatisfy=False
                    break
        if isSatisfy:#如果属于当前数据集
            total_num+=1 
            if datalist[FatherTypeIndex]==ChosenVal:#如果特征属性取值是预设值
                if datalist[-1]=='1':
                    NumList[0][0]+=1
                else:
                    NumList[0][1]+=1
            else:
                if datalist[-1]=='1':
                    NumList[1][0]+=1
                else:
                    NumList[1][1]+=1
    #计算基尼指数
    gini=0
    for i in range(2):
        temp=NumList[i][0]+NumList[i][1]
        if temp==0:
            gini+=0
        else:
            gini+=(temp/total_num)*(1-pow(NumList[i][0]/temp,2)-pow(NumList[i][1]/temp,2))
    return gini

def Build_OriTypeValList(DataLists):
    """
    建立原数据集属性取值列表
    """
    OriTypeValList=dict()
    for i in range(len(DataLists[0])-1):
        OriTypeValList[i]=[]
    for datalist in DataLists:
        for i in range(len(datalist)-1):
            if datalist[i] not in OriTypeValList[i]:
                OriTypeValList[i].append(datalist[i])
    return OriTypeValList

def Build_Tree_ID3(DataLists, OriTypeValList, TypeUsedList, TypeValList, Depth):
    """
    构建ID3决策树（基于当前节点）
    """
    total_num=0                  #当前节点样本总数量
    pos_num=0                    #当前节点样本值为1的数量
    for DataList in DataLists:
        #以下检测样本DataList是否属于当前节点
        isSatisfy=True
        for i in range(len(TypeUsedList)):
            if TypeUsedList[i]==True:
                if DataList[i] in TypeValList[i]:
                    continue
                else:
                    isSatisfy=False
                    break
        if isSatisfy:            #如果DataList属于当前节点
            total_num+=1
            if DataList[-1]=='1':#如果样本值为1
                pos_num+=1

    if Depth==len(TypeUsedList): #如果递归达到最深，即所有属性均已决策
        if 2*pos_num>=total_num:
            return 1#返回叶节点1
        else:
            return 0#返回叶节点0
    elif pos_num==0:             #如果样本值均为0，返回叶节点0
        return 0
    elif pos_num==total_num:     #如果样本值均为1，返回叶节点1
        return 1
    else:
        TypeGainDict=dict()      #字典{属性：对应信息增益}
        for i in range(len(TypeUsedList)):
            if TypeUsedList[i]:  #若该属性已决策，跳过
                continue
            #计算未决策属性的信息增益
            TypeGainDict[i]=Calculate_Gain(DataLists, TypeUsedList, TypeValList, i)
        #以下寻找当前信息增益最大的未决策属性
        MaxGain=-1
        NodeIndex=0              #最大信息增益的未决策属性索引
        for i in TypeGainDict.keys():
            if MaxGain<TypeGainDict[i]:
                MaxGain=TypeGainDict[i]
                NodeIndex=i
        TreeDict=dict()         #子决策树
        NodeValList=[]          #即将决策的属性的值列表
        #填充即将决策的属性的出现值列表
        for DataList in DataLists:
            isSatisfy=True
            for i in range(len(TypeUsedList)):
                if TypeUsedList[i]==True:
                    if DataList[i] in TypeValList[i]:
                        continue
                    else:
                        isSatisfy=False
                        break
            if isSatisfy:
                if DataList[NodeIndex] not in NodeValList:
                    NodeValList.append(DataList[NodeIndex])
        #构建子决策树的TypeUsedList
        NewTypeUsedList=copy.deepcopy(TypeUsedList)
        NewTypeUsedList[NodeIndex]=True
        #构建子决策树
        for NodeVal in OriTypeValList[NodeIndex]:
            TreeKEY=str(NodeIndex)+':'+'1'+':'+NodeVal      #键： "属性序号:是/否:属性值"
            if NodeVal in NodeValList:              #该属性值有训练集样本对应时，递归构建子决策树
                NewTypeValList=copy.deepcopy(TypeValList)
                NewTypeValList[NodeIndex].append(NodeVal)
                TreeDict[TreeKEY]=Build_Tree_ID3(DataLists, OriTypeValList, NewTypeUsedList, NewTypeValList,Depth+1)
            else:                                   #该属性值没有训练集样本对应时，构建叶节点
                if pos_num>=total_num-pos_num:
                    TreeDict[TreeKEY]=1
                else:
                    TreeDict[TreeKEY]=0
        return TreeDict

def Build_Tree_C45(DataLists, OriTypeValList, TypeUsedList, TypeValList, Depth):
    """
    构建C45决策树（基于当前节点）
    """
    total_num=0                  #当前节点样本总数量
    pos_num=0                    #当前节点样本值为1的数量
    for DataList in DataLists:
        #以下检测样本DataList是否属于当前节点
        isSatisfy=True
        for i in range(len(TypeUsedList)):
            if TypeUsedList[i]==True:
                if DataList[i] in TypeValList[i]:
                    continue
                else:
                    isSatisfy=False
                    break
        if isSatisfy:            #如果DataList属于当前节点
            total_num+=1
            if DataList[-1]=='1':#如果样本值为1
                pos_num+=1

    if Depth==len(TypeUsedList): #如果递归达到最深，即所有属性均已决策
        if 2*pos_num>=total_num:
            return 1#返回叶节点1
        else:
            return 0#返回叶节点0
    elif pos_num==0:             #如果样本值均为0，返回叶节点0
        return 0
    elif pos_num==total_num:     #如果样本值均为1，返回叶节点1
        return 1
    else:
        TypeGRDict=dict()      #字典{属性：对应信息增益}
        for i in range(len(TypeUsedList)):
            if TypeUsedList[i]:  #若该属性已决策，跳过
                continue
            #计算未决策属性的信息增益
            TypeGRDict[i]=Calculate_GainRatio(DataLists, TypeUsedList, TypeValList, i)
        #以下寻找当前信息增益最大的未决策属性
        MaxGR=-1
        NodeIndex=0              #最大信息增益的未决策属性索引
        for i in TypeGRDict.keys():
            if MaxGR<TypeGRDict[i]:
                MaxGR=TypeGRDict[i]
                NodeIndex=i
        TreeDict=dict()         #子决策树
        NodeValList=[]          #即将决策的属性的值列表
        #填充即将决策的属性的出现值列表
        for DataList in DataLists:
            isSatisfy=True
            for i in range(len(TypeUsedList)):
                if TypeUsedList[i]==True:
                    if DataList[i] in TypeValList[i]:
                        continue
                    else:
                        isSatisfy=False
                        break
            if isSatisfy:
                if DataList[NodeIndex] not in NodeValList:
                    NodeValList.append(DataList[NodeIndex])
        #构建子决策树的TypeUsedList
        NewTypeUsedList=copy.deepcopy(TypeUsedList)
        NewTypeUsedList[NodeIndex]=True
        #构建子决策树
        for NodeVal in OriTypeValList[NodeIndex]:
            TreeKEY=str(NodeIndex)+':'+'1'+':'+NodeVal      #键： "属性序号:是/否:属性值"
            if NodeVal in NodeValList:              #该属性值有训练集样本对应时，递归构建子决策树
                NewTypeValList=copy.deepcopy(TypeValList)
                NewTypeValList[NodeIndex].append(NodeVal)
                TreeDict[TreeKEY]=Build_Tree_C45(DataLists, OriTypeValList, NewTypeUsedList, NewTypeValList,Depth+1)
            else:                                   #该属性值没有训练集样本对应时，构建叶节点
                if pos_num>=total_num-pos_num:
                    TreeDict[TreeKEY]=1
                else:
                    TreeDict[TreeKEY]=0
        return TreeDict

def Build_Tree_CART(DataLists, OriTypeValList, TypeUsedList, TypeValList, Depth):
    """
    构建CART决策树（基于当前节点）
    """
    total_num=0                  #当前节点样本总数量
    pos_num=0                    #当前节点样本值为1的数量
    #统计total_num和pos_num
    for DataList in DataLists:
        #以下检测样本DataList是否属于当前节点
        isSatisfy=True
        for i in range(len(TypeUsedList)):
            if TypeUsedList[i]==True:
                if DataList[i] in TypeValList[i]:
                    continue
                else:
                    isSatisfy=False
                    break
        if isSatisfy:            #如果DataList属于当前节点
            total_num+=1
            if DataList[-1]=='1':#如果样本值为1
                pos_num+=1

    if Depth==len(TypeUsedList): #如果递归达到最深，即所有属性均已决策
        if 2*pos_num>total_num:
            return 1#返回叶节点1
        else:
            return 0#返回叶节点0
    elif pos_num==0:             #如果样本值均为0，返回叶节点0
        return 0
    elif pos_num==total_num:     #如果样本值均为1，返回叶节点1
        return 1
    else:
        TypeGINIDict=dict()      #字典{属性：对应信息增益}
        for i in range(len(TypeUsedList)):
            if TypeUsedList[i]:  #若该属性已决策，跳过
                continue
            #计算未决策属性的信息增益
            TypeGINIDict[i]=[]
            for val in OriTypeValList[i]:
                TypeGINIDict[i].append(Calculate_GINI(DataLists, TypeUsedList, TypeValList, i,val))
        #以下寻找当前最小基尼系数的未决策属性和对应的属性值
        MinGINI=-1
        NodeIndex=0              #最小基尼系数的未决策属性索引
        NodeVal='\0'             #最小基尼系数的未决策属性索引的划分值
        for i in TypeGINIDict.keys():
            for j in range(len(TypeGINIDict[i])):
                if MinGINI==-1 or MinGINI>=TypeGINIDict[i][j]:
                    MinGINI=TypeGINIDict[i][j]
                    NodeIndex=i
                    NodeVal=OriTypeValList[i][j]
        TreeDict=dict()         #子决策树
        isValExist=[False,False]          #即将决策的属性的值列表
        #填充即将决策的属性的出现值列表
        for DataList in DataLists:
            isSatisfy=True
            for i in range(len(TypeUsedList)):
                if TypeUsedList[i]==True:
                    if DataList[i] in TypeValList[i]:
                        continue
                    else:
                        isSatisfy=False
                        break
            if isSatisfy:
                if DataList[NodeIndex]==NodeVal:
                    isValExist[0]=True
                else:
                    isValExist[1]=True
            if isValExist[0] and isValExist[1]:
                break
        #构建子决策树的TypeUsedList
        NewTypeUsedList=copy.deepcopy(TypeUsedList)
        NewTypeUsedList[NodeIndex]=True
        #构建子决策树
        TreeKEY=str(NodeIndex)+':'+'1'+':'+NodeVal
        if isValExist[0]:
            NewTypeValList=copy.deepcopy(TypeValList)
            NewTypeValList[NodeIndex].append(NodeVal)
            TreeDict[TreeKEY]=Build_Tree_CART(DataLists, OriTypeValList, NewTypeUsedList, NewTypeValList,Depth+1)
        else:
            if pos_num>total_num-pos_num:
                TreeDict[TreeKEY]=1
            else:
                TreeDict[TreeKEY]=0
        TreeKEY=str(NodeIndex)+':'+'0'+':'+NodeVal
        if isValExist[1]:
            NewTypeValList=copy.deepcopy(TypeValList)
            for val in OriTypeValList[NodeIndex]:
                if val!=NodeVal:
                    NewTypeValList[NodeIndex].append(val)
            TreeDict[TreeKEY]=Build_Tree_CART(DataLists, OriTypeValList, NewTypeUsedList, NewTypeValList,Depth+1)
        else:
            if pos_num>total_num-pos_num:
                TreeDict[TreeKEY]=1
            else:
                TreeDict[TreeKEY]=0
        return TreeDict

def predict_Text(TreeDict, TextList):
    """
    对样本TextList进行预测
    """
    if type(TreeDict).__name__!='dict':
        #print(TreeDict)
        return TreeDict
    else:
        for key in TreeDict.keys():
            keylist=key.split(':')
            if TextList[int(keylist[0])]==keylist[2] and keylist[1]=='1':
                #print(key)
                return predict_Text(TreeDict[key],TextList)
            if TextList[int(keylist[0])]!=keylist[2] and keylist[1]=='0':
                #print(key)
                return predict_Text(TreeDict[key],TextList)

def predict_Valid(TreeDict, ValidTextLists):
    """
    对验证集ValidTextLists预测，返回准确率
    """
    correct=0
    for TextList in ValidTextLists:
        presult=predict_Text(TreeDict, TextList)
        if presult==int(TextList[-1]):
            correct+=1 
    return correct/len(ValidTextLists)

def predict_Test(TreeDict, TestTextLists):
    """
    对测试集TestTextLists预测，返回预测值
    """
    result=[]
    for TextList in TestTextLists:
        presult=predict_Text(TreeDict, TextList)
        result.append(presult)
    return result

#################################################################
#读取文本
with open("AIlab2/lab2_dataset/car_train.csv") as tf:
    FileText=tf.readlines()
#构建数据集
DataLists=[]
for Text in FileText[1:]:
    words=Text.split(',')
    if words[-1]=="1\n":
        words[-1]='1'
    else:
        words[-1]='0'
    DataLists.append(words)
OriTypeValList=Build_OriTypeValList(DataLists)

TypeNum=len(DataLists[0])-1
TypeUsedList=[]
TypeValList=[]
for i in range(TypeNum):
    TypeUsedList.append(False)
    TypeValList.append([])
#划分训练集数量
TrainSize=round(len(DataLists)*0.7)
#查看各类决策树准确率
ID3TreeDict=Build_Tree_ID3(DataLists[:TrainSize], OriTypeValList, TypeUsedList, TypeValList,0)
print("ID3: "+str(predict_Valid(ID3TreeDict,DataLists[TrainSize:])))
C45TreeDict=Build_Tree_C45(DataLists[:TrainSize], OriTypeValList, TypeUsedList, TypeValList,0)
print("C4.5: "+str(predict_Valid(C45TreeDict,DataLists[TrainSize:])))
CARTTreeDict=Build_Tree_CART(DataLists[:TrainSize], OriTypeValList, TypeUsedList, TypeValList,0)
print("CART: "+str(predict_Valid(CARTTreeDict,DataLists[TrainSize:])))
result=predict_Test(CARTTreeDict, DataLists[TrainSize:])
with open("predict.txt","a+") as tf:
    for num in result:
        tf.write(str(num)+'\n')
#print(TreeDict)