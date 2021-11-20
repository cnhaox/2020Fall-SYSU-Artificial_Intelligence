import pygame

class BlackChess():
    """黑棋显示对象"""
    def __init__(self, screen, x, y):
        self.screen = screen
        self.image = pygame.image.load('images/blackchess.png')
        self.rect = self.image.get_rect()
        self.screen_rect = screen.get_rect()
        self.rect.centerx=150+x*60
        self.rect.centery=150+y*60

    def blitme(self):
        self.screen.blit(self.image, self.rect)

class WhiteChess():
    """白棋显示对象"""
    def __init__(self, screen, x, y):
        self.screen = screen
        self.image = pygame.image.load('images/whitechess.png')
        self.rect = self.image.get_rect()
        self.screen_rect = screen.get_rect()
        self.rect.centerx=150+x*60
        self.rect.centery=150+y*60

    def blitme(self):
        self.screen.blit(self.image, self.rect)

class Board():
    """棋盘显示对象"""
    def __init__(self, screen):
        self.screen = screen
        self.image = pygame.image.load('images/board.png')
        self.rect = self.image.get_rect()
        self.screen_rect = screen.get_rect()
        self.rect.x=0
        self.rect.y=0

    def blitme(self):
        self.screen.blit(self.image, self.rect)

class Star():
    """星状物显示对象"""
    def __init__(self, screen, x, y):
        self.screen = screen
        self.image = pygame.image.load('images/star.png')
        self.rect = self.image.get_rect()
        self.screen_rect = screen.get_rect()
        self.rect.centerx=150+x*60
        self.rect.centery=150+y*60
        self.isShow=False
    def set_position(self,x,y):
        self.rect.centerx=150+x*60
        self.rect.centery=150+y*60
        self.isShow=True
    def blitme(self):
        self.screen.blit(self.image, self.rect)
