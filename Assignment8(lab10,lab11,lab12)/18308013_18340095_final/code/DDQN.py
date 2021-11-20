from Othello import *
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

LR, EPISODE, BATCH_SIZE = 0.001, 50000, 32
GAMMA, ALPHA, EPSILON = 0.9, 0.8, 0
POOL_SIZE = 200
UPDATE_DELAY = 10
BOARD_SIZE = 8
STATE_COUNT = BOARD_SIZE ** 2
# 五元组大小：s、s'、action、reward、is_end
RECORD_LENGTH = STATE_COUNT * 2 + 1 + 1 + 1
WHITE, BLACK = -1, 1


def board_to_onehot(x):
    x = x.flatten()
    result = np.zeros((3, STATE_COUNT))
    # 白、空、黑对应-1,0,1；加1之后正好对应one-hot的三个类别
    for idx in range(len(x)):
        result[int(x[idx]) + 1, idx] = 1
    return torch.tensor(result, dtype=torch.float)


def batch_to_onehot(batch_x):
    result = torch.zeros((BATCH_SIZE, 3, STATE_COUNT), dtype=torch.float)
    for i in range(BATCH_SIZE):
        result[i] = board_to_onehot(batch_x[i])
    return result


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv1d(3, 8, 3, 1, 1)
        self.conv2 = nn.Conv1d(8, 12, 3, 1, 1)
        self.linear = nn.Linear(12 * STATE_COUNT, (STATE_COUNT + 1))

    def forward(self, x):
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv2_out = conv2_out.view(conv2_out.shape[0], -1)
        output = self.linear(conv2_out)
        return output


class Double_DQN:
    def __init__(self, player):
        self.pool = np.zeros((POOL_SIZE, RECORD_LENGTH))
        self.record_counter = 0
        self.iteration_counter = 0
        self.train_Q, self.target_Q = DQN().to(device), DQN().to(device)
        self.player = player
        self.optimizer = torch.optim.Adam(self.train_Q.parameters(), lr=LR)
        self.criteria = nn.MSELoss()

    def greedy_choice(self, othello):
        """
        epsilon-greedy选择函数
        """
        possible_moves = othello.get_possible_moves(self.player)
        possible_moves = list(possible_moves)

        if len(possible_moves) == 0:
            return (0, 64)                 # 表示此时没有动作

        # epsilon贪心策略
        if np.random.uniform() < EPSILON:
            # 从possible_moves里面随机选一个坐标
            idx = np.random.randint(0, len(possible_moves), 1)[0]
            pos = possible_moves[idx]
        else:
            s = board_to_onehot(othello.board).view(1, 3, -1).to(device)
            action_values = self.train_Q(s)[0]
            moves_idx = [pos[0] * BOARD_SIZE + pos[1] for pos in possible_moves]
            moves_values = action_values[moves_idx]
            print(moves_idx)
            print(moves_values)
            _, idx = torch.max(moves_values, 0)
            move = moves_idx[idx]
            pos = (int(move // BOARD_SIZE), int(move % BOARD_SIZE))
        return pos

    def expand_pool(self, s, a, r, s_, is_end):
        """
        更新经验池
        """
        record = np.hstack((s.flatten(), a[0] * BOARD_SIZE + a[1], r, s_.flatten(), is_end))
        self.pool[self.record_counter % POOL_SIZE] = record
        self.record_counter += 1

    def update_network(self, oppo_train_Q, oppo_target_Q):
        """
        更新网络
        """
        self.iteration_counter += 1
        if self.iteration_counter % UPDATE_DELAY == 0:
            # 加载训练网络至目标网络
            self.target_Q.load_state_dict(self.train_Q.state_dict())
        # 从经验池中获取一个batch
        random_indices = np.random.choice(POOL_SIZE, BATCH_SIZE)
        record_batch = self.pool[random_indices, :]

        # 棋面/状态batch
        s_batch = torch.tensor(record_batch[:, :STATE_COUNT], dtype=torch.float).to(device)
        s_batch = batch_to_onehot(s_batch).to(device)
        # 动作batch
        a_batch = torch.tensor(record_batch[:, STATE_COUNT:STATE_COUNT + 1], dtype=torch.int64).to(device)
        # 奖励batch
        r_batch = torch.tensor(record_batch[:, STATE_COUNT + 1:STATE_COUNT + 2], dtype=torch.float).to(device)
        # 下一棋面/状态/对手batch
        oppo_s_batch = torch.tensor(record_batch[:, STATE_COUNT + 2:STATE_COUNT * 2 + 2], dtype=torch.float).to(device)
        oppo_s_batch = batch_to_onehot(oppo_s_batch).to(device)
        # 是否结束batch
        is_end_batch = record_batch[:, STATE_COUNT * 2 + 2]

        batch_prediction = self.train_Q(s_batch).gather(1, a_batch)
        # 得到对手train_Q中在s'处值最大的动作
        oppo_batch_moves = torch.max(oppo_train_Q(oppo_s_batch).detach(), 1)[1].view(BATCH_SIZE, 1)
        # 然后将对手target_move中该动作的值作为Q值
        batch_target = r_batch - GAMMA * oppo_target_Q(oppo_s_batch).detach().gather(1, oppo_batch_moves)

        for idx in range(BATCH_SIZE):
            if is_end_batch[idx] == 1:
                batch_target[idx] = r_batch[idx]

        self.optimizer.zero_grad()
        loss = self.criteria(batch_prediction, batch_target)
        loss.backward()         # 反向传播
        self.optimizer.step()   # 更新权重


if __name__ == '__main__':
    AI_first_network, AI_second_network = Double_DQN(BLACK), Double_DQN(WHITE)
    epoch_start = 12000
    AI_first_network.train_Q.load_state_dict(torch.load('AI_first_model_12000_v2.pth'))
    AI_second_network.train_Q.load_state_dict(torch.load('AI_second_model_12000_v2.pth'))

    for episode in range(epoch_start, EPISODE):
        othello = Othello(BOARD_SIZE, WHITE, BLACK)
        counter = 0
        if episode % 10 == 0:
            print(episode)
        while True:
            # 先手
            s = othello.board
            a_first = AI_first_network.greedy_choice(othello)
            othello.add_chess(a_first, BLACK)
            r = othello.game_over() * 50.0 * BLACK
            s_ = othello.board
            is_end = 1 if abs(r) > 0 else 0
            AI_first_network.expand_pool(s, a_first, r, s_, is_end)
            counter += 1

            if is_end:
                break
            if AI_first_network.record_counter >= BATCH_SIZE:
                AI_first_network.update_network(AI_second_network.train_Q, AI_second_network.target_Q)

            # 后手
            s = s_
            a_second = AI_second_network.greedy_choice(othello)
            othello.add_chess(a_second, WHITE)
            r = othello.game_over() * 50.0 * WHITE
            s_ = othello.board
            is_end = 1 if abs(r) > 0 else 0
            AI_second_network.expand_pool(s, a_second, r, s_, is_end)
            counter += 1

            if is_end or (a_first == (0, 64) and a_second == (0, 64)):
                break
            if AI_second_network.record_counter >= BATCH_SIZE:
                AI_second_network.update_network(AI_first_network.train_Q, AI_first_network.target_Q)

        # 每100个episode保存一次模型
        if (episode + 1) % 1000 == 0:
            print('当前episode: ', episode + 1)
            torch.save(AI_first_network.train_Q.state_dict(), 'AI_first_model_' + str(episode + 1) + '_v2.pth')
            torch.save(AI_second_network.train_Q.state_dict(), 'AI_second_model_' + str(episode + 1) + '_v2.pth')