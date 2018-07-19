import numpy as np
import tensorflow as tf


def free_squares(board):
    free = [i for i in range(9) if board[i] == 0]

    return free


class HumanPlayer(object):

    def player(self):
        return self.__class__.__name__

    @staticmethod
    def move(board):
        print("Free Squares: " + str(free_squares(board)))
        mov = input("Please choose a position from the free squares: ")

        return int(mov)


class RandomPlayer(object):

    def player(self):
        return self.__class__.__name__

    @staticmethod
    def move(board):
        free = free_squares(board)
        mov = np.random.choice(free)

        return mov


class AIPlayer(object):

    def __init__(self):
        self.input_positions = tf.placeholder(tf.float32, shape=[None, 9])
        self.labels = tf.placeholder(tf.float32, shape=[None, 9])
        #self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self._prediction = None
        self._train_step = None
        self._error = None

    def graph(self):
        return tf.Graph()

    @property
    def prediction(self):
        if not self._prediction:
            weights = tf.Variable(tf.truncated_normal([9, 9], stddev=0.1))
            bias = tf.Variable(tf.constant(0.1, shape=[9]))
            y = tf.matmul(self.input_positions, weights) + bias
            self._prediction = tf.nn.softmax(y)

        return self._prediction

    @property
    def train_step(self):
        if not self._train_step:
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.prediction,
                labels=self.labels))
            self._train_step = tf.train.GradientDescentOptimizer(
                learning_rate=0.1).minimize(cross_entropy)

        return self._train_step

    def move(self, board):
        # sess = self.infer_session()
        # sess.run(init)

        probs = sess.run(self.prediction,
                         feed_dict={self.input_positions: [board]})[0][0]
        return probs


# ai = AIPlayer()
# init = tf.global_variables_initializer()
#
# # Start training
# with tf.Session() as sess:
#
#     # Run the initializer
#     sess.run(init)
#
#     print(ai.move([1,-1,0,0,0,0,0,0,0]))

# with tf.Graph().as_default():
#
#     ai = AIPlayer()
#
#     print(ai.train_step)
#     print(ai._prediction)
#
#     print(ai.move([1,-1,0,0,0,0,0,0,0]))


# y1_tanh = tf.tanh(tf.matmul(self.input_positions * w1) + b1)

# w1 = tf.Variable(tf.truncated_normal([9, 9], stddev=0.1))
# b1 = tf.Variable(tf.zeros([1, 9]))
