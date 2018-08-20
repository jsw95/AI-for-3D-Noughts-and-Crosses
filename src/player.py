import numpy as np
import tensorflow as tf
import random


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
    def move(board):
        free = free_squares(board)
        print("Please input as: layer x-coord y-coord")
        # move = input("Please choose a position from the free squares: {}".format(board))
        # move = g.print_num()

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
        self.player = "human"

    @staticmethod
    def move(board):
        free = free_squares(board)
        move = np.random.choice(free)

        return move


class AIPlayer(object):

    def __init__(self, model):
        self.model = model

        tf.reset_default_graph()
        self.imported_meta = tf.train.import_meta_graph("../models/{}/{}.ckpt.meta".format(self.model, self.model))

    def player(self):
        return "ai"

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
        self.player = "agentRL"
        self.Qtable = {}
        self.training = training
        if not self.training:
            tf.reset_default_graph()
            self.imported_meta = tf.train.import_meta_graph("../models/{}/{}.ckpt.meta".format(self.model, self.model))

    def training_move(self, board, e, sess, prediction, q_vals, inputs):
        free = free_squares(board)

        if random.random() < e:
            move = np.random.choice(free)
            return move

        else:

            if not self.training:
                # with tf.Session() as sess:
                self.imported_meta.restore(sess, tf.train.latest_checkpoint('../models/{}/'.format(self.model)))

            pred, q_vals_out = sess.run([prediction, q_vals], feed_dict={inputs: [board]})

            # Returns the highest scoring free square
            m = [i for i in enumerate(q_vals_out[0]) if i[0] in free]
            move_scores = sorted(m, key=lambda x: x[1], reverse=True)

            return move_scores[0][0]

    def move(self, board, e):
        free = free_squares(board)

        if random.random() < e:
            move = np.random.choice(free)
            return move

        else:
            with tf.Session() as sess:

                self.imported_meta.restore(sess, tf.train.latest_checkpoint('../models/{}/'.format(self.model)))
                q_vals_out = sess.run(["q_values:0"], feed_dict={"x:0": [board]})

                # Returns free squares ranked by score
                m = [i for i in enumerate(q_vals_out[0][0]) if i[0] in free]
                move_scores = sorted(m, key=lambda x: x[1], reverse=True)

                # Chooses randomly from the top n moves
                top_n = 4
                if top_n < len(move_scores):
                    move_rank = random.randint(0, top_n)
                else:
                    move_rank = 0

                return move_scores[move_rank][0]



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

    # def update_qtable(self, board_log, board_prev_log, move_log, reward):
    #     """
    #     The main learning algorithm for Q-learning
    #     Takes in history of moves after game with reward dependant on victory
    #     Creates and updates Q-table according to algorithm
    #     """
    #     board = board_log[-1]
    #     board_prev = board_prev_log[-1]
    #     move = move_log[-1]
    #
    #     for board in board_log:
    #         if board not in self.Qtable:
    #             action_dict = {}
    #             for i in range(9):
    #                 action_dict[str(i)] = 0.01  # random.uniform(-0.1, 0.1)
    #             self.Qtable[board] = action_dict
    #
    #     for board_prev in board_prev_log:
    #         if board_prev not in self.Qtable:
    #             action_dict = {}
    #             for i in range(9):
    #                 action_dict[str(i)] = 0.01  # random.uniform(-0.1, 0.1)
    #             self.Qtable[board_prev] = action_dict
    #
    #     if board_prev is not None:
    #         q_vals = [self.Qtable[board_prev][str(a)] for a in range(9)]
    #         prediction = reward - (self.gamma * max(q_vals))
    #
    #         change = self.alpha * (prediction - self.Qtable[board][move])
    #         # self.Qtable[board_prev][move] += change
    #         for i in range(len(move_log)):
    #             self.Qtable[board_prev_log[-i]][move_log[-i]] += change * (self.gamma ** i)
    #         self.Qtable[board_prev_log[-3]][move_log[-3]] += change
    #         print("Reward: {}, max_qvals: {}".format(reward, max(q_vals)))
    #         print("Board prev: {}".format(board_prev))
    #         print("Qtable[board_prev]: {}".format(self.Qtable[board_prev]))
    #         print("Change: {},".format(change))
