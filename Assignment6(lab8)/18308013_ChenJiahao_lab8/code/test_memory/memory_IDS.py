from memory_profiler import profile

def read_Data(FilePath):
    with open(FilePath) as tf:
        FileText=tf.readlines()
    for i in range(len(FileText)):
        if FileText[i][-1]=='\n':
            FileText[i]=FileText[i][:-2]
    return FileText

class node():
    def __init__(self,X,Y,Depth):
        self.x=X
        self.y=Y
        self.depth=Depth    # 节点深度
        self.path=[(X,Y)]   # 从初始节点到当前节点的路径
    def get_address(self):
        return (self.x, self.y)
    def update_path(self,last):
        self.path=list(last.path)
        self.path.append(self.get_address())

def Depth_limited_Search(graph, StartNode, EndNode, limited_Depth):
    '''
    深度受限搜索
    '''
    stack=list()
    lenX=len(graph)
    lenY=len(graph[0])
    stack.append(StartNode)
    while len(stack)!=0:
        CurrentNode=stack.pop()
        if CurrentNode.depth>limited_Depth:# 超过最大深度，跳过
            continue
        else:
            if CurrentNode.get_address()==EndNode.get_address():# 搜索到目标节点，返回目标节点
                return CurrentNode
            else:
                # 将周围可到达的不在路径上的节点压栈
                if CurrentNode.y>0 and graph[CurrentNode.x][CurrentNode.y-1]!='1':
                    NextNode=node(CurrentNode.x,CurrentNode.y-1,CurrentNode.depth+1)
                    if NextNode.get_address() not in CurrentNode.path:# 该节点不在路径上，压栈
                        NextNode.update_path(CurrentNode)
                        stack.append(NextNode)
                if CurrentNode.x>0 and graph[CurrentNode.x-1][CurrentNode.y]!='1':
                    NextNode=node(CurrentNode.x-1,CurrentNode.y,CurrentNode.depth+1)
                    if NextNode.get_address() not in CurrentNode.path:
                        NextNode.update_path(CurrentNode)
                        stack.append(NextNode)
                if CurrentNode.x<lenX-1 and graph[CurrentNode.x+1][CurrentNode.y]!='1':
                    NextNode=node(CurrentNode.x+1,CurrentNode.y,CurrentNode.depth+1)
                    if NextNode.get_address() not in CurrentNode.path:
                        NextNode.update_path(CurrentNode)
                        stack.append(NextNode)
                if CurrentNode.y<lenY-1 and graph[CurrentNode.x][CurrentNode.y+1]!='1':
                    NextNode=node(CurrentNode.x,CurrentNode.y+1,CurrentNode.depth+1)
                    if NextNode.get_address() not in CurrentNode.path:
                        NextNode.update_path(CurrentNode)
                        stack.append(NextNode)
    return None # 不存在解，返回空值

@profile(precision=10) 
def Ierative_Deepening_Search(graph, StartNode, EndNode):
    '''
    迭代加深搜索
    '''
    lenX=len(graph)
    lenY=len(graph[0])
    for i in range(lenX*lenY):
        finish=Depth_limited_Search(graph,StartNode,EndNode,i)
        if finish is None:# 该深度没有解
            continue
        else:             # 该深度有解
            #print("Success! The length is "+str(finish.depth))
            #print(finish.path)
            return True
    #print("Fail!")
    return False

def main():
    FileText=read_Data('MazeData.txt')
    for i in range(len(FileText)):
        for j in range(len(FileText[i])):
            if FileText[i][j]=='S':
                StartNode=node(i,j,0)
            elif FileText[i][j]=='E':
                EndNode=node(i,j,0)
    Ierative_Deepening_Search(FileText, StartNode, EndNode)

if __name__=='__main__':
    main()

