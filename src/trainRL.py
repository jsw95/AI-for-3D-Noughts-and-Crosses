# from player import *
# from ttt import Game
# import csv
import random
import numpy as np
from collections import defaultdict


def free_squares(board):
    free = [i for i in range(9) if board[i] == 0]

    return free


class AgentRL(object):

    def __init__(self, alpha, gamma, epsilon, player_number):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.player_number = player_number
        self.first_move = True
        self.Qtable = {}


    def move(self, board):
        self.first_move = False
        free = free_squares(board)
        # print("Free squares {} in board - {}".format(free, board))
        if random.random() < self.epsilon:
            move = np.random.choice(free)
            return move

        else:
            prev = self.get_qvalues(str(board))
            move = max([i for i in sorted(prev, key=prev.get) if int(i) in free])
            return move

    def get_qvalues(self, board):
        if str(board) not in self.Qtable:
            action_dict = {}
            for i in range(9):
                action_dict[str(i)] = random.uniform(-0.1, 0.1)
            self.Qtable[str(board)] = action_dict
        return self.Qtable[str(board)]

    # def get_reward(self, prev_board, prev_move):
    #     if not first_move: # update qtable
    #         prev_qvals = self.get_qvalues(prev_board, prev_move)
    #         max_qval = max

    def print_Qtable(self):
        [print(i) for i in self.Qtable.items()]


    def update_qtable(self, board, board_prev, action, reward):
        """
        The main learning algorithm for Q-learning

        """

        if str(board) not in self.Qtable:
            action_dict = {}
            for i in range(9):
                action_dict[str(i)] = random.uniform(-0.1, 0.1)
            self.Qtable[str(board)] = action_dict

        if str(board_prev) not in self.Qtable:
            action_dict = {}
            for i in range(9):
                action_dict[str(i)] = random.uniform(-0.1, 0.1)
            self.Qtable[str(board_prev)] = action_dict
        #
        # if board_prev is not None:
        #     q_vals = [self.Qtable[board_prev][str(a)] for a in range(9)]
        #     self.Qtable[board][action] = (1 - self.alpha) * self.Qtable[board][action] + \
        #         self.alpha * (reward + self.gamma * max(q_vals))
        # else:
        #     self.Qtable[board][action] = (1 - self.alpha) * self.Qtable[board][action] + \
        #         self.alpha * reward

        if board_prev is not None:
            q_vals = [self.Qtable[board_prev][str(a)] for a in range(9)]
            prediction = reward + (self.gamma * max(q_vals))

            change = self.alpha * (prediction - self.Qtable[board][action])
            self.Qtable[board][action] += change
            # print(self.Qtable[board][action])
            # print("qvals:{}".format(q_vals))
            # print(action)
        else:
            self.Qtable[board][action] += 0
            # self.Qtable[board][action] = (1 - self.alpha) * self.Qtable[board][action] + \
            #     self.alpha * reward






#
#                               agent = AgentRL(0.1, 0.2, 0.3, 1)
# agent.get_Qvalues([0, 0, 0, 0, 0, 0, 0, 0, 0], 3)
# print(agent.move([0, 0, 0, 0, 0, 0, 0, 0, 0]))
#
#
# game = Game(agent, 2)
# print(game.winners)
# print(game.advance(4))
#
# print(game.end)

# print(Game.end)