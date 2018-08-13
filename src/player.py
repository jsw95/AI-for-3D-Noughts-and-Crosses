import numpy as np
import tensorflow as tf


def free_squares(board):
    free = [i for i in range(9) if board[i] == 0]

    return free


class HumanPlayer(object):

    def __init__(self):
        self.player = "human"

    @staticmethod
    def move(board):
        print("Please input as: layer x-coord y-coord")
        coords = input("Please choose a position from the free squares: ")

        layer, x, y = coords.split(" ")


        move = (int(layer) - 1) * 16 + (int(y) - 1) * 4 + (int(x) - 1)

        return move


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
            # self.imported_meta.restore(sess, tf.train.latest_checkpoint('/tmp/'))

            # graph = tf.get_default_graph()
            # logits = graph.get_tensor_by_name('logits:0')
            # x = graph.get_tensor_by_name('x:0')
            # self.imported_meta.restore(sess, "../models/{}.ckpt.data-00000-of-00001".format(self.model))
            # weights = sess.run(logits, feed_dict={x: [board]})[0]



            weights = sess.run('logits:0', feed_dict={'x:0': [board]})[0]

            free = free_squares(board=board)

            m = [i for i in enumerate(weights) if i[0] in free]

            move_scores = sorted(m, key=lambda x: x[1], reverse=True)

            return move_scores[0][0]

# a = AIPlayer(model="3d-100-1-10-15-0.5")
# print()
a = HumanPlayer()
print(a.player)