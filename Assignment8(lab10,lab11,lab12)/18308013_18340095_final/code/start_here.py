from Othello import *
from DDQN import *
import time
import pygame

BOARD_SIZE, BOARD_WIDTH = 8, 400
GRID_WIDTH = BOARD_WIDTH / (BOARD_SIZE + 2)
WHITE, BLACK = -1, 1
background = pygame.image.load(r'1.jpg')


def update_board(board, screen, valid_moves=[], last_chess=None):
    screen.blit(background, (0, 0))

    # 绘制列
    x = 0
    while x <= BOARD_SIZE * GRID_WIDTH:
        x += GRID_WIDTH
        pygame.draw.line(screen, (0, 0, 0), (x, GRID_WIDTH), (x, GRID_WIDTH * (BOARD_SIZE + 1)))

    # 绘制行
    y = 0
    while y <= BOARD_SIZE * GRID_WIDTH:
        y += GRID_WIDTH
        pygame.draw.line(screen, (0, 0, 0), (GRID_WIDTH, y), (GRID_WIDTH * (BOARD_SIZE + 1), y))

    for row in range(board.shape[0]):
        for col in range(board.shape[1]):
            center = (int((col + 1.5) * GRID_WIDTH), int((row + 1.5) * GRID_WIDTH))
            if board[row, col] == WHITE:
                pygame.draw.circle(screen, (255, 255, 255), center, int(GRID_WIDTH / 3))
            elif board[row, col] == BLACK:
                pygame.draw.circle(screen, (0, 0, 0), center, int(GRID_WIDTH / 3))

    # print('len2: ', len(valid_moves))
    for pos in valid_moves:
        center = (int((pos[1] + 1.5) * GRID_WIDTH), int((pos[0] + 1.5) * GRID_WIDTH))
        pygame.draw.circle(screen, (0, 0, 255), center, int(GRID_WIDTH / 3), 1)
    # print('finished')

    if isinstance(last_chess, tuple):
        center = (int((last_chess[1] + 1.5) * GRID_WIDTH), int((last_chess[0] + 1.5) * GRID_WIDTH))
        pygame.draw.circle(screen, (255, 0, 0), center, int(GRID_WIDTH / 8))

    pygame.display.flip()


def main():
    AI_first = int(input('Please input 1 for AI-first mode or 0 for User-first mode:'))

    AI = BLACK if AI_first else WHITE
    HUMAN = WHITE if AI_first else BLACK
    next_player = AI if AI_first else HUMAN

    screen = pygame.display.set_mode((BOARD_WIDTH, BOARD_WIDTH))
    pygame.display.set_caption('Othello')
    othello = Othello(BOARD_SIZE, WHITE, BLACK)
    update_board(othello.board, screen)

    Network = Double_DQN(AI)
    if AI_first:
        Network.train_Q.load_state_dict(torch.load('AI_first_model_22000_v2.pth'))
    else:
        Network.train_Q.load_state_dict(torch.load('AI_second_model_30000_v2.pth'))
    othello = Othello(BOARD_SIZE, WHITE, BLACK)

    if_quit = False
    pos = (-1, -1)
    while True:
        cur_state = othello.game_over()
        if cur_state == AI:
            print('AI wins!')
            break
        elif cur_state == HUMAN:
            print('HUMAN wins!')
            break

        if next_player == HUMAN:
            valid_moves = othello.get_possible_moves(next_player)
            update_board(othello.board, screen, valid_moves, pos)

            if len(valid_moves) == 0:
                print('Player ', 'AI' if next_player == AI else 'HUMAN', ' has no valid move...')
                time.sleep(2)
                next_player = - next_player
                continue

            while next_player == HUMAN:
                for event in pygame.event.get():            # 轮询事件
                    if event.type == pygame.QUIT:           # 点击了关闭窗口
                        if_quit = True
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        pos = (int(event.pos[1] / GRID_WIDTH) - 1, int(event.pos[0] / GRID_WIDTH) - 1)
                        if pos in valid_moves:
                            othello.add_chess(pos, HUMAN)
                            update_board(othello.board, screen, last_chess=pos)
                            next_player = - next_player
                            time.sleep(1)
                            break
            AI_chess = len(othello.white_chess) if AI == WHITE else len(othello.black_chess)
            HUMAN_chess = len(othello.white_chess) if HUMAN == WHITE else len(othello.black_chess)
            print('HUMAN finished;', ' AI chess: ', AI_chess, '; HUMAN chess: ', HUMAN_chess)

            if if_quit:
                break

        # AI
        valid_moves = othello.get_possible_moves(next_player)
        if len(valid_moves) == 0:
            print('Player ', 'AI' if next_player == AI else 'HUMAN', ' has no valid move...')
            time.sleep(2)
            next_player = - next_player
            continue

        pos = Network.greedy_choice(othello)
        othello.add_chess(pos, AI)
        update_board(othello.board, screen, valid_moves, pos)
        # time.sleep(1)
        next_player = - next_player
        print('AI choice: ', pos)
        AI_chess = len(othello.white_chess) if AI == WHITE else len(othello.black_chess)
        HUMAN_chess = len(othello.white_chess) if HUMAN == WHITE else len(othello.black_chess)
        print('AI finished;', ' AI chess: ', AI_chess, '; HUMAN chess: ', HUMAN_chess)

    time.sleep(1000)


if __name__ == '__main__':
    main()
