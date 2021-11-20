import math
def build_WordDict(FileText):
    """
    构建TF-IDF字典
    """
    WordDict=dict()                 #字典
    TextSize=len(FileText)          #文档数量
    TextIndex=0
    WordNumPerSen=[0]*TextSize      #文档单词数量列表
    #构建词频字典
    for text in FileText:
        TextList=text.split('\t')   #按\t分割文本
        words=TextList[2].split()   #将句子分割成单词列表
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

with open("semeval.txt") as f:
    FileText=f.readlines()
WordDict=build_WordDict(FileText)
print(WordDict['goal'][1])
print(WordDict['delight'][1])
print(WordDict['for'][1])
print(WordDict['sheva'][1])
#with open("18308013_ChenJiahao_TFIDF.txt",'a+') as f:
#    for i in range(len(FileText)):
#        isFirst=True
#        for word in sorted(WordDict.keys()):
#            if isFirst:
#                isFirst=False
#            else:
#                f.write(' ')
#            f.write(str(WordDict[word][i]))
#        f.write('\n')
