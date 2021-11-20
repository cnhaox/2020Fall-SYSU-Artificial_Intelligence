from memory_profiler import profile

class node():
    def __init__(self,X,Y,G):
        self.x=X
        self.y=Y
        self.g=G
        self.h=0
        self.path=[(X,Y)]
    def __lt__(self,other):     # 重载<，根据f值排序
        return self.g+self.h<other.g+other.h
    def get_address(self):
        return (self.x, self.y)
    def update_path(self,last):
        self.path=list(last.path)
        self.path.append(self.get_address())
    def update_h(self,end,D):   # 使用曼哈顿距离
        self.h=D*abs(self.x-end.x)+abs(self.y-end.y)

def read_Data(FilePath):
    with open(FilePath) as tf:
        FileText=tf.readlines()
    for i in range(len(FileText)):
        if FileText[i][-1]=='\n':
            FileText[i]=FileText[i][:-2]
    return FileText

@profile(precision=10) 
def A_star_search(graph,StartNode,EndNode):
    '''
    A*搜索
    '''
    frontier=PriorityQueue()# 待探索节点组成的优先队列
    explored=set()          # 已探索节点组成的集合
    lenX=len(graph)
    lenY=len(graph[0])
    frontier.put(StartNode)
    while True:
        if frontier.empty():# 队列无待探索的节点，搜索失败
            #print('Fail!')
            return False
        else:
            CurrentNode=frontier.get()
            if CurrentNode.get_address()==EndNode.get_address():# 搜索到目标节点，成功
                #print("Success! The length is "+str(CurrentNode.g))
                #print(CurrentNode.path)
                return True
            if CurrentNode.get_address() in explored:# 当前节点已被探索过，跳过
                continue
            else:
                explored.add(CurrentNode.get_address())
            #  将周围可到达的节点加入队列
            if CurrentNode.y>0 and graph[CurrentNode.x][CurrentNode.y-1]!='1':
                NextNode=node(CurrentNode.x,CurrentNode.y-1,CurrentNode.g+1)
                NextNode.update_path(CurrentNode)
                NextNode.update_h(EndNode,1)
                frontier.put(NextNode)
            if CurrentNode.x>0 and graph[CurrentNode.x-1][CurrentNode.y]!='1':
                NextNode=node(CurrentNode.x-1,CurrentNode.y,CurrentNode.g+1)
                NextNode.update_path(CurrentNode)
                NextNode.update_h(EndNode,1)
                frontier.put(NextNode)
            if CurrentNode.x<lenX-1 and graph[CurrentNode.x+1][CurrentNode.y]!='1':
                NextNode=node(CurrentNode.x+1,CurrentNode.y,CurrentNode.g+1)
                NextNode.update_path(CurrentNode)
                NextNode.update_h(EndNode,1)
                frontier.put(NextNode)
            if CurrentNode.y<lenY-1 and graph[CurrentNode.x][CurrentNode.y+1]!='1':
                NextNode=node(CurrentNode.x,CurrentNode.y+1,CurrentNode.g+1)
                NextNode.update_path(CurrentNode)
                NextNode.update_h(EndNode,1)
                frontier.put(NextNode)


def main():
    FileText=read_Data('MazeData.txt')
    for i in range(len(FileText)):
        for j in range(len(FileText[i])):
            if FileText[i][j]=='S':
                StartNode=node(i,j,0)
            elif FileText[i][j]=='E':
                EndNode=node(i,j,0)
    StartNode.update_h(EndNode,1)
    A_star_search(FileText,StartNode,EndNode)
            


if __name__=='__main__':
    from queue import Queue, PriorityQueue
    main()