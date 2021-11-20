class node():
    def __init__(self,X,Y,G):
        self.x=X
        self.y=Y
        self.g=G
        self.h=0
        self.path=[(X,Y)]
    def get_f(self):
        return self.g+self.h
    def __lt__(self,other):
        return self.g+self.h<other.g+other.h
    def get_address(self):
        return (self.x, self.y)
    def update_path(self,last):
        self.path=list(last.path)
        self.path.append(self.get_address())
    def update_h(self,end,D):
        self.h=D*abs(self.x-end.x)+abs(self.y-end.y)

def read_Data(FilePath):
    with open(FilePath) as tf:
        FileText=tf.readlines()
    for i in range(len(FileText)):
        if FileText[i][-1]=='\n':
            FileText[i]=FileText[i][:-2]
    return FileText


def Search(graph, StartNode, EndNode, limits):
    '''
    f值受限搜索
    '''
    frontier=PriorityQueue()
    lenX=len(graph)
    lenY=len(graph[0])
    frontier.put(StartNode)
    while not frontier.empty():
        CurrentNode=frontier.get()
        if CurrentNode.get_f()>limits:  # 当前节点f值超过limits，直接退出
            return None
        else:
            if CurrentNode.get_address()==EndNode.get_address():# 搜到目标节点，返回该节点
                return CurrentNode
            else:
                #  将周围可到达的不在路径上的节点加入队列
                if CurrentNode.y>0 and graph[CurrentNode.x][CurrentNode.y-1]!='1':
                    NextNode=node(CurrentNode.x,CurrentNode.y-1,CurrentNode.g+1)
                    if NextNode.get_address() not in CurrentNode.path:
                        NextNode.update_path(CurrentNode)
                        NextNode.update_h(EndNode,1)
                        frontier.put(NextNode)
                if CurrentNode.x>0 and graph[CurrentNode.x-1][CurrentNode.y]!='1':
                    NextNode=node(CurrentNode.x-1,CurrentNode.y,CurrentNode.g+1)
                    if NextNode.get_address() not in CurrentNode.path:
                        NextNode.update_path(CurrentNode)
                        NextNode.update_h(EndNode,1)
                        frontier.put(NextNode)
                if CurrentNode.x<lenX-1 and graph[CurrentNode.x+1][CurrentNode.y]!='1':
                    NextNode=node(CurrentNode.x+1,CurrentNode.y,CurrentNode.g+1)
                    if NextNode.get_address() not in CurrentNode.path:
                        NextNode.update_path(CurrentNode)
                        NextNode.update_h(EndNode,1)
                        frontier.put(NextNode)
                if CurrentNode.y<lenY-1 and graph[CurrentNode.x][CurrentNode.y+1]!='1':
                    NextNode=node(CurrentNode.x,CurrentNode.y+1,CurrentNode.g+1)
                    if NextNode.get_address() not in CurrentNode.path:
                        NextNode.update_path(CurrentNode)
                        NextNode.update_h(EndNode,1)
                        frontier.put(NextNode)
    return None # 队列为空，无解，返回空值

def IDA_Star_Search(graph, StartNode, EndNode):
    '''
    IDA*搜索
    '''
    lenX=len(graph)
    lenY=len(graph[0])
    for i in range(lenX*lenY):
        finish=Search(graph,StartNode,EndNode,i)
        if finish is None:
            continue
        else:
            #print("Success! The length is "+str(finish.g))
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
    StartNode.update_h(EndNode,1)
    IDA_Star_Search(FileText, StartNode, EndNode)
            


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
    StartNode.update_h(EndNode,1)
    timer1=Timer("IDA_Star_Search(FileText, StartNode, EndNode)","from __main__ import IDA_Star_Search, FileText, StartNode, EndNode")
    total_time=timer1.timeit(500)
    print("IDA* after 500 times: "+str(total_time)+"secs")
    #main()