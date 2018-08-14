import numpy as np
from player import RandomPlayer
from player import HumanPlayer
from trainRL import AgentRL
import random
import matplotlib.pyplot as plt


class Game(object):

    def __init__(self, agent, player):
        self.agent = agent
        self.player = player
        self.player_current = 1
        self.winner = None
        self.end = False
        self.board = [0] * 9
        self.board_log = []
        self.move_log = []
        self.print = False
        self.total_reward = 0
        self.agent_wins = 0
        self.random_wins = 0
        self.draws = 0

    winners = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6]
    ]

    def win_check(self):
        for win in self.winners:
            if (self.board[win[0]] == self.board[win[1]]
                    and self.board[win[0]] == self.board[win[2]]
                    and self.board[win[0]] != 0):
                self.winner = self.player_current
                self.end = True
                return True

    def draw_check(self):
        if 0 not in self.board:
            self.end = True
            return True  # returns true for draw but no winner

    # def move(self, pos):
    #     if pos < 0 or pos > 8:
    #         print("Please enter a square between 0 and 8")
    #         # self.move(pos=pos)
    #
    #     elif self.board[pos] != 0:
    #         print("Please choose a blank square")
    #         # self.move(pos=pos)
    #     else:
    #         self.board[pos] = self.player_current

    def reset_game(self):
        self.board = [0] * 9
        self.player_current = 1
        self.winner = None
        self.end = False

    def print_board(self):
        def f(l):
            if l == 0:
                return ' '
            elif l == 1:
                return 'X'
            elif l == -1:
                return 'O'

        print('\n-------------')
        print('| {} | {} | {} |'.format(f(self.board[0]), f(self.board[1]), f(self.board[2])))
        print('-------------')
        print('| {} | {} | {} |'.format(f(self.board[3]), f(self.board[4]), f(self.board[5])))
        print('-------------')
        print('| {} | {} | {} |'.format(f(self.board[6]), f(self.board[7]), f(self.board[8])))
        print('-------------\n')

    def free_squares(self):
        free = [i for i in range(9) if self.board[i] == 0]
        return free

    def play(self, agent, player, agent_first=1, e=0.1, print=False):
        """
        Main game playing function. Each call is one game.
        Can choose random or human player against an agent.
        Agent Q-table is updated after each game based on reward.
        """

        # Storing history of moves and boards (only relevant ones)
        board_log, board_prev_log, move_log = [], [], []


        if agent_first < 0.5:  # Who moves first
            move = player.move(self.board)
            self.board[int(move)] = self.player_current  # update board with move
            self.player_current *= -1
            if print:
                self.print_board()

        while True:  # Training loop
            if print:
                self.print_board()

            board_prev = str(self.board)
            move = agent.move(self.board, epsilon=e)  # Agents move
            self.board[int(move)] = self.player_current
            board = str(self.board)

            move_log.append(str(move))
            board_prev_log.append(board_prev)
            board_log.append(board)

            if self.win_check():
                reward = 1
                self.agent_wins += 1
                self.total_reward += reward
                break

            if self.draw_check():
                reward = 0
                self.draws += 1
                break

            self.player_current *= -1

            r_move = player.move(self.board)
            self.board[r_move] = self.player_current

            if self.win_check():
                reward = -1
                self.random_wins += 1
                self.total_reward += reward
                break

            if self.draw_check():
                reward = 0
                self.draws += 1
                break

            self.player_current *= -1

        agent.update_qtable(board_log, board_prev_log, move_log, reward)
        self.reset_game()


# Training and testing -- move to new file

p1 = AgentRL(0.01, 0.9, 0.9)
p2 = RandomPlayer()
p3 = HumanPlayer()
reward_log = []
game = Game(p1, p2)

iter = 5000
for i in range(iter):
    n = random.random()
    if i%10000 == 0:
        print("\nGame number: {}".format(i))
    game.play(p1, p2, agent_first=n, e=(iter-i+1)/iter)
    reward_log.append(game.total_reward)


game.agent_wins = 0
game.random_wins = 0
game.draws = 0
for i in range(10000):
    n = random.random()
    if i%1000 == 10:
        print("\nGame number: {}".format(i))
    game.play(p1, p2, agent_first=n, e=0)

print("Agent wins: {}".format(game.agent_wins))
print("Draws: {}".format(game.draws))
print("Random wins: {}".format(game.random_wins))
print(p1.Qtable["[0, 0, 0, 0, 0, 0, 0, 0, 0]"])


for i in range(5):
    n = random.random()
    game.play(p1, p3, agent_first=n, e=0, print=True)

