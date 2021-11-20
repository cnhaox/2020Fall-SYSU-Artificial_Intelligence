import sys
import pygame
import numpy as np
from settings import Settings
from image import BlackChess, WhiteChess, Board, Star
from chess import Chess

def check_events(settings, screen, chess, star):
    """响应按键和鼠标事件"""
    global WChessList# 白棋显示对象列表
    global BChessList# 黑棋显示对象列表
    global ScoreList # 分数显示列表
    global step      # 当前步数
    global NumList  # 序号显示列表
    for event in pygame.event.get():
        if event.type == pygame.QUIT:       # 关闭
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE: # 按下空格键：显示局面分数
                settings.ShowScore = True
            elif event.key == pygame.K_q:   # 按下q键：退出
                sys.exit()
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE: # 松开空格键
                if settings.ShowScore:
                    chess.displayScore()
                settings.ShowScore=False
        elif event.type == pygame.MOUSEBUTTONDOWN:# 检测按下鼠标
            if settings.isFinish!=0:           # 游戏结束后不再响应
                return
            if settings.movement==False:    # 只响应一次点击
                if settings.HumanSize!=settings.CurrentPlayer:# 当前下棋方不是人类时不响应
                    continue
                mouse_x, mouse_y=event.pos  # 获取鼠标点击位置
                if mouse_x>=780 or mouse_x<120 or mouse_y>=780 or mouse_y<120:
                    continue                # 点击位置不在识别范围内时不响应
                # 将鼠标点击位置映射到落子位置
                IX=int((mouse_x-120)/60)
                IY=int((mouse_y-120)/60)
                if chess.chessboard[IY,IX]!=0:
                    continue                # 落子位置已有子，不响应
                chess.chessboard[IY,IX]=settings.CurrentPlayer# 落子
                NumList.append((IX, IY))
                ScoreList[step%2]=chess.evaluate()
                step+=1
                if chess.judge_win(IY, IX, settings.CurrentPlayer):# 判断是否赢了
                    settings.isFinish=settings.CurrentPlayer
                # 切换下棋方
                if settings.CurrentPlayer==1:
                    BChessList.append(BlackChess(screen,IX,IY))# 创建黑棋显示对象
                    star.set_position(IX,IY)
                    settings.CurrentPlayer=2
                else:
                    WChessList.append(WhiteChess(screen,IX,IY))# 创建白棋显示对象
                    star.set_position(IX,IY)
                    settings.CurrentPlayer=1
                # 判断棋盘是否无子可下、到达平局
                isFinishT=True
                for i in range(chess.size):
                    for j in range(chess.size):
                        if chess.chessboard[i,j]==0:
                            isFinishT=False
                            break
                    if isFinishT==False:
                        break
                if isFinishT and settings.isFinish==0:
                    settings.isFinish=-1

            settings.movement=True
        elif event.type == pygame.MOUSEBUTTONUP:# 检测松开鼠标
            settings.movement=False

def main():
    """游戏主程序"""
    # 初始化信息
    str = input("你想执黑/白？：0.我想看电脑下；1.黑；2.白\n请输入数字(0/1/2)：")
    if str[0] == '0':
        HumanSize=0
    elif str[0]=='1':
        HumanSize=1
    else:
        HumanSize=2
    str = input("选择难度？\n请输入数字(1~9)：")
    MaxDepth=ord(str[0])-ord('0')
    if MaxDepth<0 or MaxDepth>9:
        MaxDepth=2
    # 初始化五子棋数据
    chess=Chess(11,HumanSize,MaxDepth)
    # 初始化游戏并创建一个屏幕对象
    pygame.init()
    game_settings=Settings(HumanSize)
    screen = pygame.display.set_mode((game_settings.screen_width, game_settings.screen_width))
    pygame.display.set_caption("五子棋")
    board=Board(screen)
    global WChessList# 白棋显示对象列表
    WChessList=list()
    global BChessList# 黑棋显示对象列表
    BChessList=list()
    global ScoreList # 分数显示列表
    ScoreList=[chess.evaluate(),0]
    global step      # 当前步数
    step=0
    global NumList  # 序号显示列表
    NumList=list()
    star=Star(screen, 0, 0)# 星型显示对象

    # 为棋盘上已有的棋创建显示对象
    for i in range(chess.size):
        for j in range(chess.size):
            if chess.chessboard[i,j]==1:
                BChessList.append(BlackChess(screen,j,i))
            elif chess.chessboard[i,j]==2:
                WChessList.append(WhiteChess(screen,j,i))
    game_font=pygame.font.SysFont('宋体',36,True)
    # 开始游戏的主循环
    while True:
        # 继续下棋
        if game_settings.isFinish==0:
            if game_settings.CurrentPlayer!=game_settings.HumanSize:# 轮到电脑下棋
                # 获取最优位置
                CBestScore, BestLocation=chess.search(game_settings.CurrentPlayer,-float('inf'),float('inf'),1)
                chess.displayScore()
                if BestLocation==None:# 没有位置可下，平局
                    game_settings.isFinish=-1
                else:
                    chess.chessboard[BestLocation[0],BestLocation[1]]=game_settings.CurrentPlayer# 落子
                    NumList.append((BestLocation[1],BestLocation[0]))
                    ScoreList[step%2]=chess.evaluate()
                    step+=1
                    # 判断胜利
                    if chess.judge_win(BestLocation[0],BestLocation[1], game_settings.CurrentPlayer):
                        game_settings.isFinish=game_settings.CurrentPlayer
                    # 换边
                    if game_settings.CurrentPlayer==1:
                        BChessList.append(BlackChess(screen,BestLocation[1],BestLocation[0]))
                        star.set_position(BestLocation[1],BestLocation[0])
                        game_settings.CurrentPlayer=2
                    elif game_settings.CurrentPlayer==2:
                        WChessList.append(WhiteChess(screen,BestLocation[1],BestLocation[0]))
                        star.set_position(BestLocation[1],BestLocation[0])
                        game_settings.CurrentPlayer=1
            
        # 监视键盘和鼠标事件
        check_events(game_settings, screen, chess, star)
        #显示
        board.blitme()
        for wc in WChessList:
            wc.blitme()
        for bc in BChessList:
            bc.blitme()
        # if star.isShow:
        #     star.blitme()
        if step==1:
            screen.blit(game_font.render('AI predicted score after Step %d: %f' %(step, ScoreList[(step-1)%2]), True, [200, 200, 200]),[30,10])
        elif step>1:
            screen.blit(game_font.render('AI predicted score after Step %d: %f' %(step-1, ScoreList[(step-2)%2]), True, [200, 200, 200]),[30,10])
            screen.blit(game_font.render('AI predicted score after Step %d: %f' %(step, ScoreList[(step-1)%2]), True, [200, 200, 200]),[30,40])
        if game_settings.isFinish==1:
            screen.blit(game_font.render('Black Win!', True, [255, 0, 0]),[150,70])
        elif game_settings.isFinish==2:
            screen.blit(game_font.render('White Win!', True, [255, 0, 0]),[150,70])
        elif game_settings.isFinish==-1:
            screen.blit(game_font.render('Draw!!', True, [255, 0, 0]),[150,70])
        index=1
        for (x,y) in NumList:
            screen.blit(game_font.render('%d' % index, True, [255, 0, 0]),[135+x*60,135+y*60])
            index+=1
        # 让最近绘制的屏幕可见
        pygame.display.flip()

if __name__=='__main__':
    main()