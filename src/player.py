import numpy as np
import tensorflow as tf


def free_squares(board):
    free = [i for i in range(9) if board[i] == 0]

    return free

class HumanPlayer(object):

    def __init__(self):
        self.human = True

    def player(self):
        return self.__class__.__name__



    @staticmethod
    def move(board):

        print("Free Squares: " + str(free_squares(board)))
        move = input("Please choose a position from the free squares: ")

        return int(move)



class RandomPlayer(object):

    def __init__(self):
        self.human = False

    def player(self):
        return self.__class__.__name__

    @staticmethod
    def move(board):
        free = free_squares(board)
        move = np.random.choice(free)

        return move



class AIPlayer(object):

    def __init__(self):
        tf.reset_default_graph()
        self.imported_meta = tf.train.import_meta_graph("/tmp/model_2d.ckpt.meta")
        self.human = False

    def move(self,  board):

        with tf.Session() as sess:

            self.imported_meta.restore(sess, tf.train.latest_checkpoint('/tmp/'))

            weights = sess.run('logits:0', feed_dict={'x:0': [board]})[0]

            free = free_squares(board=board)

            d = [i for i in enumerate(weights) if i[0] in free]

            move_scores = sorted(d, key=lambda x: x[1], reverse=True)

            return move_scores[0][0]






