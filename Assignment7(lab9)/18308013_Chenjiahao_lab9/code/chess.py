import numpy as np
import random
class Chess():
    def __init__(self,N, ChosenSize, MaxDepth):
        self.size=N # 棋盘边尺寸 
        self.chessboard = np.zeros((N,N), dtype=np.int)# 棋盘
        self.ChessScore = np.zeros((N,N))# 各个位置的分数表

        self.initChessboard()       # 初始化棋盘
        self.HumanSize=ChosenSize   # 人类选边
        self.MaxDepth=MaxDepth      # 最大深度


    def initChessboard(self):
        """初始化棋盘"""
        CenterIndex = int(self.size/2)
        self.chessboard[CenterIndex, CenterIndex]=1
        if CenterIndex+1<self.size:
            self.chessboard[CenterIndex, CenterIndex+1]=1# 黑棋标记1
            self.chessboard[CenterIndex+1, CenterIndex]=2# 白棋标记2
        if CenterIndex-1>0:
            self.chessboard[CenterIndex, CenterIndex-1]=2
        return

    def getQuintupleScore(self, num):
        """五元组分数表"""
        tempScore=0
        if num==1:
            tempScore=10
        elif num==2:
            tempScore=30
        elif num==3:
            tempScore=120
        elif num==4:
            tempScore=450
        elif num==5:
            tempScore=10000
        return tempScore

    def judge_win(self, x, y, CurrentPlayer):
        """判断[x,y]处的落子是否导致胜利"""
        if self.chessboard[x,y]!=CurrentPlayer:
            return False
        # 判断列
        num=1
        tempx=x-1
        tempy=y
        while tempx>0 and num<5:
            if self.chessboard[tempx,tempy]==CurrentPlayer:
                num+=1
                tempx-=1
            else:
                break
        tempx=x+1
        while tempx<self.size and num<5:
            if self.chessboard[tempx,tempy]==CurrentPlayer:
                num+=1
                tempx+=1
            else:
                break
        if num>=5:
            return True
        # 判断行
        num=1
        tempx=x
        tempy=y-1
        while tempy>0 and num<5:
            if self.chessboard[tempx,tempy]==CurrentPlayer:
                num+=1
                tempy-=1
            else:
                break
        tempy=y+1
        while tempy<self.size and num<5:
            if self.chessboard[tempx,tempy]==CurrentPlayer:
                num+=1
                tempy+=1
            else:
                break
        if num>=5:
            return True
        # 判断/
        num=1
        tempx=x+1
        tempy=y-1
        while tempx<self.size and tempy>0 and num<5:
            if self.chessboard[tempx,tempy]==CurrentPlayer:
                num+=1
                tempx+=1
                tempy-=1
            else:
                break
        tempx=x-1
        tempy=y+1
        while tempx>0 and tempy<self.size and num<5:
            if self.chessboard[tempx,tempy]==CurrentPlayer:
                num+=1
                tempx-=1
                tempy+=1
            else:
                break
        if num>=5:
            return True
        # 判断\
        num=1
        tempx=x+1
        tempy=y+1
        while tempx<self.size and tempy<self.size and num<5:
            if self.chessboard[tempx,tempy]==CurrentPlayer:
                num+=1
                tempx+=1
                tempy+=1
            else:
                break
        tempx=x-1
        tempy=y-1
        while tempx>0 and tempy>0 and num<5:
            if self.chessboard[tempx,tempy]==CurrentPlayer:
                num+=1
                tempx-=1
                tempy-=1
            else:
                break
        if num>=5:
            return True
        return False

        
    def evaluate(self):
        """计算返回当前棋面的效益值"""
        score=0 # 总效益值
        # 按行判断
        for i in range(self.size):
            Xnum=0
            Onum=0
            for j in range(self.size):
                if self.chessboard[i,j]==1:
                    Xnum+=1
                elif self.chessboard[i,j]==2:
                    Onum+=1
                if j<4:
                    continue
                elif j>4:
                    if self.chessboard[i,j-5]==1:
                        Xnum-=1
                    elif self.chessboard[i,j-5]==2:
                        Onum-=1
                if Xnum>0 and Onum>0:
                    score+=0
                else:
                    score+=self.getQuintupleScore(Xnum)
                    score-=self.getQuintupleScore(Onum)
        # 按列判断
        for j in range(self.size):
            Xnum=0
            Onum=0
            for i in range(self.size):
                if self.chessboard[i,j]==1:
                    Xnum+=1
                elif self.chessboard[i,j]==2:
                    Onum+=1
                if i<4:
                    continue
                elif i>4:
                    if self.chessboard[i-5,j]==1:
                        Xnum-=1
                    elif self.chessboard[i-5,j]==2:
                        Onum-=1
                if Xnum>0 and Onum>0:
                    score+=0
                else:
                    score+=self.getQuintupleScore(Xnum)
                    score-=self.getQuintupleScore(Onum)
        # 按 / 判断
        for j in range(4,self.size):
            i=0
            Xnum=0
            Onum=0
            while j>=0 and i<self.size:
                if self.chessboard[i,j]==1:
                    Xnum+=1
                elif self.chessboard[i,j]==2:
                    Onum+=1
                if i<4:
                    i+=1
                    j-=1
                    continue
                elif i>4:
                    if self.chessboard[i-5,j+5]==1:
                        Xnum-=1
                    elif self.chessboard[i-5,j+5]==2:
                        Onum-=1
                if Xnum>0 and Onum>0:
                    score+=0
                else:
                    score+=self.getQuintupleScore(Xnum)
                    score-=self.getQuintupleScore(Onum)
                i+=1
                j-=1
        for i in range(1,self.size-4):
            j=self.size-1
            Xnum=0
            Onum=0
            while j>=0 and i<self.size:
                if self.chessboard[i,j]==1:
                    Xnum+=1
                elif self.chessboard[i,j]==2:
                    Onum+=1
                if j>self.size-5:
                    i+=1
                    j-=1
                    continue
                elif j<self.size-5:
                    if self.chessboard[i-5,j+5]==1:
                        Xnum-=1
                    elif self.chessboard[i-5,j+5]==2:
                        Onum-=1
                if Xnum>0 and Onum>0:
                    score+=0
                else:
                    score+=self.getQuintupleScore(Xnum)
                    score-=self.getQuintupleScore(Onum)
                i+=1
                j-=1
        # 按 \ 判断
        for i in range(0,self.size-4):
            j=0
            Xnum=0
            Onum=0
            while i<self.size and j<self.size:
                if self.chessboard[i,j]==1:
                    Xnum+=1
                elif self.chessboard[i,j]==2:
                    Onum+=1
                if j<4:
                    i+=1
                    j+=1
                    continue
                elif j>4:
                    if self.chessboard[i-5,j-5]==1:
                        Xnum-=1
                    elif self.chessboard[i-5,j-5]==2:
                        Onum-=1
                if Xnum>0 and Onum>0:
                    score+=0
                else:
                    score+=self.getQuintupleScore(Xnum)
                    score-=self.getQuintupleScore(Onum)
                i+=1
                j+=1
        for j in range(1,self.size-4):
            i=0
            Xnum=0
            Onum=0
            while i<self.size and j<self.size:
                if self.chessboard[i,j]==1:
                    Xnum+=1
                elif self.chessboard[i,j]==2:
                    Onum+=1
                if i<4:
                    i+=1
                    j+=1
                    continue
                elif i>4:
                    if self.chessboard[i-5,j-5]==1:
                        Xnum-=1
                    elif self.chessboard[i-5,j-5]==2:
                        Onum-=1
                if Xnum>0 and Onum>0:
                    score+=0
                else:
                    score+=self.getQuintupleScore(Xnum)
                    score-=self.getQuintupleScore(Onum)
                i+=1
                j+=1
        return score
                
    def has_neighbor(self,x,y):
        """判断[x,y]附近是否有棋"""
        bound=1
        for i in range(x-bound,x+bound+1):
            for j in range(y-bound,y+bound+1):
                if i<0 or i>=self.size or j<0 or j>=self.size:
                    continue
                if i==x and j==y:
                    continue
                if self.chessboard[i,j]!=0:
                    return True
        return False


    def search(self, CurrentPlayer, alpha, beta, CurrentDepth):
        """博弈树搜索"""
        if CurrentDepth>self.MaxDepth:# 超过最大深度，叶节点
            return self.evaluate(), None
        # 初始化BestScore, BestLocation
        if CurrentPlayer==1:
            BestScore=-float('inf')
        else:
            BestScore=float('inf')
        BestLocation=None
        Talpha=alpha
        Tbeta=beta
        # 遍历所有位置
        for i in range(self.size):
            # if CurrentDepth==1:
            #     print(str(i)+' begin')
            for j in range(self.size):
                if CurrentDepth==1:
                    self.ChessScore[i,j]=float('nan')
                if self.chessboard[i,j]==0 and self.has_neighbor(i,j):# 只考虑周围有棋的空位置
                    self.chessboard[i,j]=CurrentPlayer# 落子
                    if self.judge_win(i,j,CurrentPlayer):# 出现胜利，叶节点
                        self.chessboard[i,j]=0# 收子
                        if CurrentPlayer==1:      # 落子方为黑棋
                            if CurrentDepth==1:
                                self.ChessScore[i,j]=1000000/CurrentDepth# 除以深度是为了尽早胜利
                            BestScore=1000000/CurrentDepth
                            BestLocation=(i,j)
                        elif CurrentPlayer==2:    # 落子方为白棋
                            if CurrentDepth==1:
                                self.ChessScore[i,j]=-1000000/CurrentDepth
                            BestScore=-1000000/CurrentDepth
                            BestLocation=(i,j)
                        return BestScore, BestLocation
                    elif CurrentPlayer==1:      # 落子方为黑棋
                        # 搜索下一个节点
                        SonScore, NextLocation=self.search(2, Talpha, Tbeta, CurrentDepth+1)
                        if CurrentDepth==1:
                            self.ChessScore[i,j]=SonScore
                        # 更新最优效益值与对应位置
                        if SonScore>BestScore:
                            BestScore=SonScore
                            BestLocation=(i,j)
                        # alpha剪枝
                        if SonScore>Talpha:
                            Talpha=SonScore
                        if Talpha>Tbeta:
                            self.chessboard[i,j]=0
                            return BestScore, BestLocation
                    elif CurrentPlayer==2:    # 落子方为白棋
                        # 搜索下一个节点
                        SonScore, NextLocation=self.search(1, Talpha, Tbeta, CurrentDepth+1)
                        if CurrentDepth==1:
                            self.ChessScore[i,j]=SonScore
                        # 更新最优效益值与对应位置
                        if SonScore<BestScore:
                            BestScore=SonScore
                            BestLocation=(i,j)
                        # beta剪枝
                        if SonScore<Tbeta:
                            Tbeta=SonScore
                        if Talpha>Tbeta:
                            self.chessboard[i,j]=0
                            return BestScore, BestLocation
                    self.chessboard[i,j]=0# 收子
        if BestLocation==None:
            BestScore=0
        return BestScore, BestLocation

    def display(self):
        """通过命令行显示棋盘"""
        for i in range(self.size):
            for j in range(self.size):
                print("----", end="")
            print("-")
            for j in range(self.size):
                print("| ", end="")
                if self.chessboard[i,j]==1:
                    print("X ", end="")
                elif self.chessboard[i,j]==2:
                    print("O ", end="")
                else:
                    print("  ", end="")
            print("|")
        for j in range(self.size):
            print("----", end="")
        print("-")
        return 

    def displayScore(self):
        """通过命令行显示各个点的分数"""
        for i in range(self.size):
            for j in range(self.size):
                print(str(self.ChessScore[i,j])+'\t',end="")
            print("")
        print("")
        return 
