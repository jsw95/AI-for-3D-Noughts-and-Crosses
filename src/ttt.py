import numpy as np
from player import RandomPlayer
from player import HumanPlayer
from player import AgentRL
import random
import matplotlib.pyplot as plt
import tensorflow as tf


class Game(object):

    def __init__(self, agent, player):
        self.agent = agent
        self.player = player
        self.player_current = 1
        self.winner = None
        self.board = [0] * 9
        self.total_reward = 0
        self.agent_wins = 0
        self.random_wins = 0
        self.winners_back1 = self.winners_back1()
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
                return True

    def draw_check(self):
        if 0 not in self.board:
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

    def winners_back1(self):
        winners_back1 = []
        for win in self.winners:
            for p in range(len(win)):
                winners_back1.append(([x for i, x in enumerate(win) if i != p], win[p]))
        return winners_back1

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

    def bin_board(self, board):
        b_p1 = [1 if i == 1 else 0 for i in board]
        b_p2 = [1 if i == -1 else 0 for i in board]
        bin_board = b_p1 + b_p2
        return bin_board

    # One hot encoding, used for converting moves
    @staticmethod
    def ohe(i):
        enc = [0] * 9
        enc[i] = 1
        return enc

    def play(self, agent, player, sess=None, q_vals=None, inputs=None,
             agent_first=1, e=0, print_data=False):
        """
        Main game playing function. Each call is one game.
        Can choose random or human player against an agent.
        Returns game history of states, actions and rewards
        """

        self.reset_game()

        # Storing history of moves and boards (only relevant ones)
        board_log, board_prev_log, move_log = [], [], []

        if agent_first < 0.5:  # Who moves first

            # determines who other player is. needed for agent epsilon value
            if player.player == "agentRL":
                move_ = player.move(self.bin_board([i * self.player_current for i in self.board]), e=player.epsilon)

            elif player.player == "smart":
                move_ = player.smart_move(self.board, self.winners_back1)

            else:
                move_ = player.move(self.board)

            self.board[int(move_)] = self.player_current  # update board with move
            self.player_current *= -1

            if print_data:
                self.print_board()

        while True:  # Training loop
            if print_data:
                self.print_board()

            if agent.deep:
                board_prev = self.bin_board([i * self.player_current for i in self.board])
            else:
                board_prev = str(self.board)  # For q-table

            if agent.training is False:
                if agent.deep:
                    move = agent.move(board_prev, e)
                else:
                    move = agent.move(self.board, e)
            else:
                move = agent.training_move(board_prev, e, sess, q_vals, inputs) # Agents move if training

            self.board[int(move)] = self.player_current

            if agent.deep:
                board = self.bin_board([i * self.player_current for i in self.board])  # * by current player to ensure same point of view
                move = self.ohe(move)
            else:
                board = self.board  # for q-table

            # Adding the (s', a, s, r) to game history
            move_log.append(move)
            board_prev_log.append(board_prev)
            board_log.append(board)

            if self.win_check():
                reward = 1
                self.agent_wins += 1
                break
            elif self.draw_check():
                reward = 0
                self.draws += 1
                break

            self.player_current *= -1
            if print_data:
                self.print_board()

            # determines who other player is. needed for agent epsilon value
            if player.player == "agentRL":
                move_ = player.move(self.bin_board([i * self.player_current for i in self.board]), e=player.epsilon)

            elif player.player == "smart":
                move_ = player.smart_move(self.board, self.winners_back1)

            else:
                move_ = player.move(self.board)  # RandomPlayer or HumanPlayer move

            self.board[move_] = self.player_current

            if self.win_check():
                reward = -1
                self.random_wins += 1
                break
            elif self.draw_check():
                reward = 0
                self.draws += 1
                break

            self.player_current *= -1

        if agent.deep:
            reward_log = [0] * len(move_log)
            reward_log[-1] = reward

            return board_prev_log, move_log, board_log, reward_log

        else:
            agent.update_qtable(board_log, board_prev_log, move_log, reward)  # uncomment for q-table


