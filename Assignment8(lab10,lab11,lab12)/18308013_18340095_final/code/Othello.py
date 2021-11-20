import numpy as np


class Othello:
    def __init__(self, board_size, white, black):
        self.board_size, self.white, self.black = board_size, white, black
        self.board = self.reset_board()
        self.black_chess, self.white_chess = set(), set()

        self.white_chess.add((self.board_size // 2 - 1, self.board_size // 2 - 1))
        self.black_chess.add((self.board_size // 2 - 1, self.board_size // 2))
        self.black_chess.add((self.board_size // 2, self.board_size // 2 - 1))
        self.white_chess.add((self.board_size // 2, self.board_size // 2))

    def reset_board(self):
        board = np.zeros((self.board_size, self.board_size), dtype=np.int)
        board[self.board_size // 2 - 1, self.board_size // 2 - 1] = self.white
        board[self.board_size // 2 - 1, self.board_size // 2] = self.black
        board[self.board_size // 2, self.board_size // 2 - 1] = self.black
        board[self.board_size // 2, self.board_size // 2] = self.white
        return board

    def can_put(self, player, pos, dx, dy):
        x, y, step = pos[0] + dx, pos[1] + dy, 1
        while 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x, y] == - player:
            x, y = x + dx, y + dy
            step += 1

        if 0 <= x < self.board_size and 0 <= y < self.board_size and \
                (not self.board[x, y]) and step >= 2:
            return True, (x, y)
        else:
            return False, None

    def game_over(self):
        # 游戏没结束或者平局
        if len(self.black_chess) == len(self.white_chess):
            return 0
        # 棋盘下满了
        if len(self.black_chess) + len(self.white_chess) == self.board_size ** 2:
            return self.white if len(self.white_chess) > len(self.black_chess) else self.black
        # 棋盘上只有一种颜色的棋子
        if len(self.black_chess) == 0 or len(self.white_chess) == 0:
            return self.white if len(self.black_chess) == 0 else self.black
        return 0

    def get_possible_moves(self, player):
        valid_moves = set()
        my_chess = self.white_chess if player == self.white else self.black_chess
        for pos in my_chess:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    is_valid, valid_pos = self.can_put(player, pos, dx, dy)
                    if is_valid:
                        valid_moves.add(valid_pos)
        return valid_moves

    def reverse_direction(self, player, start, dx, dy):
        end_x, end_y, step = start[0] + dx, start[1] + dy, 1

        # 从pos往当前方向遍历，知道遇到不是异色的位置（为空或者放着同色棋子）
        while 0 <= end_x < self.board_size and 0 <= end_y < self.board_size and self.board[end_x, end_y] == - player:
            end_x, end_y = end_x + dx, end_y + dy
            step += 1

        # 如果终点为同色棋子，就把起点到终点之间的异色棋子反转
        if 0 <= end_x < self.board_size and 0 <= end_y < self.board_size and \
                self.board[end_x, end_y] == player and step >= 2:
            cur_x, cur_y = start[0] + dx, start[1] + dy
            if player == self.white:
                while cur_x != end_x or cur_y != end_y:
                    self.board[cur_x, cur_y] = self.white
                    self.white_chess.add((cur_x, cur_y))
                    self.black_chess.remove((cur_x, cur_y))
                    cur_x, cur_y = cur_x + dx, cur_y + dy
            else:
                while cur_x != end_x or cur_y != end_y:
                    self.board[cur_x, cur_y] = self.black
                    self.black_chess.add((cur_x, cur_y))
                    self.white_chess.remove((cur_x, cur_y))
                    cur_x, cur_y = cur_x + dx, cur_y + dy

    def add_chess(self, pos, player):
        if pos == (0, 64):
            return
        self.board[pos[0], pos[1]] = player
        self.black_chess.add(pos) if player == self.black else self.white_chess.add(pos)

        # 遍历8个方向，每个方向需要则反转，否则不用反转
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                self.reverse_direction(player, pos, dx, dy)
