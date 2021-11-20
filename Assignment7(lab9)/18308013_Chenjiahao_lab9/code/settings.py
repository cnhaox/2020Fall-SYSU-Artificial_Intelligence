class Settings():
    def __init__(self, humansize):
        # 屏幕数据
        self.screen_width = 900
        self.screen_height = 900
        # 五子棋相关数据
        self.CurrentPlayer=1    # 黑棋先手
        self.HumanSize = humansize# 人类所执的棋
        self.movement=False     # 用于判断人类下棋动作
        self.isFinish=0         # 游戏结束标志
