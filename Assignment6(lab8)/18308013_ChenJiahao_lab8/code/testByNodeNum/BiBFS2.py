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

def BiBFS(graph, StartNode, EndNode):
    '''
    双向宽度优先搜索
    '''
    frontier=[]
    frontier.append(Queue())# 正向搜索待搜索队列
    frontier.append(Queue())# 反向搜索待搜索队列
    explored=[]
    explored.append([])# 正向搜索已搜索列表
    explored.append([])# 反向搜索已搜索列表
    lenX=len(graph)
    lenY=len(graph[0])
    explored_num=0
    stored_num=2
    frontier[0].put(StartNode)
    frontier[1].put(EndNode)
    Index=0
    NextIndex=(Index+1)%2
    while True:
        if frontier[Index].empty():# 当前队列为空，搜索失败
            print('Fail!')
            return False
        else:
            CurrentNode=frontier[Index].get()
            explored_num+=1
            # 判断当前节点是否已被另一搜索搜过
            for OtherNode in explored[NextIndex]:
                if CurrentNode.get_address()==OtherNode.get_address():
                    # 已被另一搜索搜过，搜索成功
                    print("Success! The length is "+str(CurrentNode.depth+OtherNode.depth))
                    if Index==0:
                        print(CurrentNode.path)
                        print(OtherNode.path[::-1])
                    else:
                        print(OtherNode.path)
                        print(CurrentNode.path[::-1])
                    print("explored: "+str(explored_num))
                    print("MAX stored: "+str(stored_num))
                    return True
            # 判断当前节点是否已被当前搜索搜过
            flag=False
            for OtherNode in explored[Index]:
                if CurrentNode.get_address()==OtherNode.get_address():
                    flag=True
                    break
            if flag:# 已被搜过，跳过
                continue
            else:   # 未被搜过，加入已搜索列表
                explored[Index].append(CurrentNode)
            # 将周围可到达的节点加入队列
            if CurrentNode.y>0 and graph[CurrentNode.x][CurrentNode.y-1]!='1':
                NextNode=node(CurrentNode.x,CurrentNode.y-1,CurrentNode.depth+1)
                NextNode.update_path(CurrentNode)
                frontier[Index].put(NextNode)
            if CurrentNode.x>0 and graph[CurrentNode.x-1][CurrentNode.y]!='1':
                NextNode=node(CurrentNode.x-1,CurrentNode.y,CurrentNode.depth+1)
                NextNode.update_path(CurrentNode)
                frontier[Index].put(NextNode)
            if CurrentNode.x<lenX-1 and graph[CurrentNode.x+1][CurrentNode.y]!='1':
                NextNode=node(CurrentNode.x+1,CurrentNode.y,CurrentNode.depth+1)
                NextNode.update_path(CurrentNode)
                frontier[Index].put(NextNode)
            if CurrentNode.y<lenY-1 and graph[CurrentNode.x][CurrentNode.y+1]!='1':
                NextNode=node(CurrentNode.x,CurrentNode.y+1,CurrentNode.depth+1)
                NextNode.update_path(CurrentNode)
                frontier[Index].put(NextNode)
            
            if stored_num<frontier[Index].qsize()+frontier[NextIndex].qsize():
                stored_num=frontier[Index].qsize()+frontier[NextIndex].qsize()
            # 切换至另一个搜索
            Index=(Index+1)%2
            NextIndex=(Index+1)%2

def main():
    FileText=read_Data('MazeData.txt')
    for i in range(len(FileText)):
        for j in range(len(FileText[i])):
            if FileText[i][j]=='S':
                StartNode=node(i,j,0)
            elif FileText[i][j]=='E':
                EndNode=node(i,j,0)
    BiBFS(FileText,StartNode,EndNode)
            


if __name__=='__main__':
    from queue import Queue, PriorityQueue
    main()