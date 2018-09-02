import numpy as np
import tensorflow as tf
import random
import pickle

def free_squares(board):
    free = [i for i in range(len(board)) if board[i] == 0]

    return free


class FreeSquareError(Exception):
    """Error for choice not being a free square"""
    pass


class HumanPlayer(object):
    def __init__(self):
        self.player = "human"

    @staticmethod
    def move(board):  # 4x4x4 currently
        free = free_squares(board)
        print([i + 1 for i in free])
        if len(board) == 9:
            print("Please input your move 1-9")
            while True:
                try:
                    move = int(input()) - 1
                    if move in free:
                        break
                    else:
                        raise FreeSquareError

                except ValueError:
                    print("Invalid entry. Try again...")
                except FreeSquareError:
                    print("Your choice is not a free square. Try again...")

            return int(move)

        elif len(board) == 64:
            print("Please input as: layer x-coord y-coord")
            while True:
                try:
                    coords = input()
                    layer, x, y = coords.split(" ")
                    move = (int(layer) - 1) * 16 + (int(y) - 1) * 4 + (int(x) - 1)
                    if move in free:
                        break
                    else:
                        raise FreeSquareError

                except ValueError:
                    print("Invalid entry. Try again...")
                except FreeSquareError:
                    print("Your choice is not a free square. Try again...")

            return int(move)


class RandomPlayer(object):
    def __init__(self):
        self.player = "random"

    @staticmethod
    def move(board):
        free = free_squares(board)
        move = np.random.choice(free)

        return move

class SmartPlayer(object):
    def __init__(self):
        self.player = "smart"


    def smart_move(self, board, winners_back1, e):
        """This function returns a move that will either block or complete a winning sequence"""
        free = free_squares(board)
        n = random.random()
        if n > 0.1:
            if len(board) == 9:
                for win in winners_back1:
                    if (board[win[0][0]] == board[win[0][1]]
                            and board[win[0][0]] != 0
                            and win[1] in free):
                        move = win[1]
                        return move

            if len(board) == 64:
                for win in winners_back1:
                    if (board[win[0][0]] == board[win[0][1]]
                            and board[win[0][0]] == board[win[0][2]]
                            and board[win[0][0]] != 0
                            and win[1] in free):
                        move = win[1]
                        return move

        move = np.random.choice(free)
        return move


class AIPlayer(object):
    def __init__(self, model):
        self.model = model
        self.player = "ai"

        tf.reset_default_graph()
        self.imported_meta = tf.train.import_meta_graph("../models/{}/{}.ckpt.meta".format(self.model, self.model))

    def move(self, board):
        with tf.Session() as sess:
            self.imported_meta.restore(sess, tf.train.latest_checkpoint('../models/{}/'.format(self.model)))

            weights = sess.run('logits:0', feed_dict={'x:0': [board]})[0]

            free = free_squares(board=board)

            m = [i for i in enumerate(weights) if i[0] in free]

            move_scores = sorted(m, key=lambda x: x[1], reverse=True)

            return move_scores[0][0]


class AgentRL(object):
    def __init__(self, epsilon, model, training=True):
        self.epsilon = epsilon
        self.model = model
        self.deep = True
        self.player = "agentRL"
        self.training = training
        if not self.training:
            tf.reset_default_graph()
            self.imported_meta = tf.train.import_meta_graph("../models/{}/{}.ckpt.meta".format(self.model, self.model))

    def to_bin(self, board):
        b_p1 = [1 if i == 1 else 0 for i in board]
        b_p2 = [1 if i == -1 else 0 for i in board]
        bin_board = b_p1 + b_p2
        return bin_board

    def free_bin(self, board):
        b1 = [i for i in board[:int(len(board)/2)]]
        b2 = [i for i in board[int(len(board)/2):]]
        free = [i for i in range(int(len(board)/2)) if (b1[i] == 0 and b2[i] == 0)]
        return free


    def training_move(self, board, e, sess, q_vals, inputs):
        """move function used when training algorithm"""
        # free = free_squares(board)
        free = self.free_bin(board)

        if random.random() < e:
            move = np.random.choice(free)
            return move

        else:
            q_vals_out = sess.run([q_vals], feed_dict={inputs: [board]})

            # Returns the highest scoring free square
            m = [i for i in enumerate(q_vals_out[0][0]) if i[0] in free]

            move_scores = sorted(m, key=lambda x: x[1], reverse=True)

            return move_scores[0][0]

    def move(self, board, e):
        free = self.free_bin(board)

        """move function used when not training"""
        # free = free_squares(board)

        if random.random() < e:  # epsilon based random move choice. e=1 gives purely random
            move = np.random.choice(free)
            return move

        else:
            with tf.Session() as sess:
                self.imported_meta.restore(sess, tf.train.latest_checkpoint('../models/{}/'.format(self.model)))
                q_vals_out = sess.run(["q_values:0"], feed_dict={"x:0": [board]})

                # Returns free squares ranked by score
                m = [i for i in enumerate(q_vals_out[0][0]) if i[0] in free]
                move_scores = sorted(m, key=lambda x: x[1], reverse=True)

                return move_scores[0][0]




class AgentRLTable(object):

    def __init__(self, alpha, gamma, epsilon, model=None, training=False):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Qtable = {}
        self.deep = False
        self.training = training
        self.player = "agentRL"
        self.model = model

        if self.model:
            with open("../models/{}".format(self.model), "rb") as handle:
                b = pickle.load(handle)

            self.Qtable = b


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
    #
    # def print_qtable(self):
    #     [print(i) for i in self.Qtable.items()]
    #
    def update_qtable(self, board_log, board_prev_log, move_log, reward):
        """
        The main learning algorithm for tabular Q-learning
        Takes in history of moves after game with reward dependant on victory
        Creates and updates Q-table according to algorithm
        """
        board = board_log[-1]
        board_prev = board_prev_log[-1]
        move = move_log[-1]

        for board in board_log:
            board = str(board)
            if board not in self.Qtable:
                action_dict = {}
                for i in range(9):
                    action_dict[str(i)] = 0.01  # random.uniform(-0.1, 0.1)
                self.Qtable[board] = action_dict

        for board_prev in board_prev_log:
            board_prev = str(board_prev)
            if board_prev not in self.Qtable:
                action_dict = {}
                for i in range(9):
                    action_dict[str(i)] = 0.01  # random.uniform(-0.1, 0.1)
                self.Qtable[board_prev] = action_dict

        if board_prev is not None:

            q_vals = [self.Qtable[str(board_prev)][str(a)] for a in range(9)]
            prediction = reward - (self.gamma * max(q_vals))

            change = self.alpha * (prediction - self.Qtable[str(board)][str(move)])
            self.Qtable[str(board_prev)][str(move)] += change

