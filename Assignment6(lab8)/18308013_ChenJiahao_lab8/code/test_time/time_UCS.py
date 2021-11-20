def read_Data(FilePath):
    with open(FilePath) as tf:
        FileText=tf.readlines()
    for i in range(len(FileText)):
        if FileText[i][-1]=='\n':
            FileText[i]=FileText[i][:-2]
    return FileText

class node():
    '''
    节点类
    '''
    def __init__(self,X,Y,Cost):
        self.x=X            # 坐标x
        self.y=Y            # 坐标y
        self.cost=Cost      # 从初始节点到当前节点的代价
        self.path=[(X,Y)]   # 从初始节点到当前节点的路径
    def __lt__(self,other): # 重载<，用于优先队列
        return self.cost<other.cost
    def get_address(self):    # 返回当前坐标的元组
        return (self.x, self.y)
    def update_path(self,last):
        self.path=list(last.path)
        self.path.append(self.get_address())

def UCS(graph, StartNode, EndNode):
    '''
    一致代价搜索
    graph：图列表，StartNode：初始节点；EndNode：目标节点
    '''
    frontier=PriorityQueue()# 等待探索的节点优先队列
    explored=set()          # 已经探索过的节点集合
    lenX=len(graph)
    lenY=len(graph[0])
    frontier.put(StartNode)
    while True:
        if frontier.empty():# 没有可探索的节点，搜索失败
            # print('Fail!')
            return False
        else:
            CurrentNode=frontier.get()# 获得队列中cost最小的节点
            if CurrentNode.get_address()==EndNode.get_address():# 是目标节点，搜索成功
                # print("Success! The length is "+str(CurrentNode.cost))
                # print(CurrentNode.path)
                return True
            if CurrentNode.get_address() in explored:# 已经探索过，跳过
                continue
            else:
                explored.add(CurrentNode.get_address())
            # 将周围可到达的节点加入队列
            if CurrentNode.y>0 and graph[CurrentNode.x][CurrentNode.y-1]!='1':
                NextNode=node(CurrentNode.x,CurrentNode.y-1,CurrentNode.cost+1)
                NextNode.update_path(CurrentNode)
                frontier.put(NextNode)
            if CurrentNode.x>0 and graph[CurrentNode.x-1][CurrentNode.y]!='1':
                NextNode=node(CurrentNode.x-1,CurrentNode.y,CurrentNode.cost+1)
                NextNode.update_path(CurrentNode)
                frontier.put(NextNode)
            if CurrentNode.x<lenX-1 and graph[CurrentNode.x+1][CurrentNode.y]!='1':
                NextNode=node(CurrentNode.x+1,CurrentNode.y,CurrentNode.cost+1)
                NextNode.update_path(CurrentNode)
                frontier.put(NextNode)
            if CurrentNode.y<lenY-1 and graph[CurrentNode.x][CurrentNode.y+1]!='1':
                NextNode=node(CurrentNode.x,CurrentNode.y+1,CurrentNode.cost+1)
                NextNode.update_path(CurrentNode)
                frontier.put(NextNode)

def main():
    FileText=read_Data('AIlab8/MazeData.txt')
    for i in range(len(FileText)):
        for j in range(len(FileText[i])):
            if FileText[i][j]=='S':
                StartNode=node(i,j,0)
            elif FileText[i][j]=='E':
                EndNode=node(i,j,0)
    UCS(FileText,StartNode,EndNode)

if __name__=='__main__':
    from queue import Queue, PriorityQueue
    from timeit import Timer
    FileText=read_Data('MazeData.txt')
    for i in range(len(FileText)):
        for j in range(len(FileText[i])):
            if FileText[i][j]=='S':
                StartNode=node(i,j,0)
            elif FileText[i][j]=='E':
                EndNode=node(i,j,0)
    #main()
    timer1=Timer("UCS(FileText,StartNode,EndNode)","from __main__ import UCS, FileText, StartNode, EndNode")
    total_time=timer1.timeit(500)
    print("UCS after 500 times: "+str(total_time)+"secs")