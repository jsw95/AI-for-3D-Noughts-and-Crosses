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

    def __init__(self, alpha, gamma, epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Qtable = {}

    def move(self, board, epsilon):
        free = free_squares(board)
        if random.random() < epsilon:
            move = np.random.choice(free)
            return move

        else:
            prev = self.get_qvalues(str(board))
            move = [i for i in sorted(prev, key=prev.get, reverse=True) if int(i) in free][0]
            return move

    def get_qvalues(self, board):
        """
        Returns action dict for input board.
        Creates if none
        """
        if str(board) not in self.Qtable:
            action_dict = {}
            for i in range(9):
                action_dict[str(i)] = 0.01  # random.uniform(-0.1, 0.1)
            self.Qtable[str(board)] = action_dict
        return self.Qtable[str(board)]

    def print_qtable(self):
        [print(i) for i in self.Qtable.items()]

    def update_qtable(self, board_log, board_prev_log, move_log, reward):
        """
        The main learning algorithm for Q-learning
        Takes in history of moves after game with reward dependant on victory
        Creates and updates Q-table according to algorithm
        """
        board = board_log[-1]
        board_prev = board_prev_log[-1]
        move = move_log[-1]

        for board in board_log:
            if board not in self.Qtable:
                action_dict = {}
                for i in range(9):
                    action_dict[str(i)] = 0.01  # random.uniform(-0.1, 0.1)
                self.Qtable[board] = action_dict

        for board_prev in board_prev_log:
            if board_prev not in self.Qtable:
                action_dict = {}
                for i in range(9):
                    action_dict[str(i)] = 0.01  # random.uniform(-0.1, 0.1)
                self.Qtable[board_prev] = action_dict

        if board_prev is not None:
            q_vals = [self.Qtable[board_prev][str(a)] for a in range(9)]
            prediction = reward - (self.gamma * max(q_vals))

            change = self.alpha * (prediction - self.Qtable[board][move])
            # self.Qtable[board_prev][move] += change
            for i in range(len(move_log)):
                self.Qtable[board_prev_log[-i]][move_log[-i]] += change * (self.gamma ** i)
            # self.Qtable[board_prev_log[-3]][move_log[-3]] += change
            # print("Reward: {}, max_qvals: {}".format(reward, max(q_vals)))
            # print("Board prev: {}".format(board_prev))
            # print("Qtable[board_prev]: {}".format(self.Qtable[board_prev]))
            # print("Change: {},".format(change))

        # else:
        #     # print("no board prev")
        #     # self.Qtable[board][action] += 0
        #     self.Qtable[board_prev][move] = (1 - self.alpha) * self.Qtable[board][action] + \
        #         self.alpha * reward
