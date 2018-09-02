import random
import numpy as np
import tensorflow as tf
from player import *
from ttt_3D import Game
# from ttt import Game
import matplotlib.pyplot as plt
import csv


def free_squares(board):
    free = [i for i in range(len(board)) if board[i] == 0]

    return free
#
def bin_board(board):
    b_p1 = [1 if i == 1 else 0 for i in board]
    b_p2 = [1 if i == -1 else 0 for i in board]
    bin_board = b_p1 + b_p2
    return bin_board

def free_bin(board):
    b1 =[i for i in board[:int(len(board)/2)]]
    b2 =[i for i in board[int(len(board)/2):]]
    free = [i for i in range(int(len(board)/2)) if (b1[i] == 0 and b2[i] == 0)]
    return free


def pass_net(input_shape, hidden_nodes, layers):
    """Neural network architecture"""

    def weight_variable(shape):
        initial = tf.truncated_normal(shape=shape, stddev=0.05)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.05, shape=shape)
        return tf.Variable(initial)

    input_positions = tf.placeholder("float", shape=[1, input_shape * 2], name="x")

    W_layer1 = weight_variable([input_shape * 2, hidden_nodes])
    W_layerM = weight_variable([hidden_nodes, hidden_nodes])
    W_layer3 = weight_variable([hidden_nodes, input_shape])
    W_layerT = weight_variable([input_shape*2, input_shape])

    b_layer1 = bias_variable([hidden_nodes])
    b_layerM = bias_variable([hidden_nodes])
    b_layer3 = bias_variable([input_shape])


    h_layer1 = tf.nn.tanh(tf.matmul(input_positions, W_layer1) + b_layerM)
    h_layer2 = tf.nn.tanh(tf.matmul(h_layer1, W_layerM) + b_layerM)
    h_layer3 = tf.nn.tanh(tf.matmul(h_layer2, W_layerM) + b_layerM)

    out_layer = tf.nn.tanh(tf.matmul(h_layer1, W_layer3) + b_layer3, name="q_values")
    if layers == 1:
        output = tf.matmul(h_layer1, W_layer3) + b_layer3

    elif layers == 2:
        output = tf.tanh(tf.matmul(h_layer2, W_layer3) + b_layer3, name="q_values")

    elif layers == 3:
        output = tf.tanh(tf.matmul(h_layer3, W_layer3) + b_layer3, name="q_values")

    # test = tf.nn.tanh(tf.matmul(input_positions, W_layerT) + b_layer3, name="q_values")
    # return input_positions, test

    # y = tf.nn.softmax(output, name="q_values")
    # o = tf.Variable(output, name="q_values")
    return input_positions, output


def trainRL(agent, player, iters, epochs, write=False, shape=64, layers=1):
    """
    Deep Q-Learning training algorithm
    Trains model to estimate weights of moves based on a board input
    """

    tf.reset_default_graph()

    inputs, q_vals = pass_net(shape, 300, layers=layers)

    # q_vals_target = tf.placeholder(tf.float32, shape=[None, shape])
    q_vals_target = tf.placeholder(tf.float32, shape=[shape,])
    loss = tf.reduce_sum(tf.square(q_vals_target - q_vals))
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    t_loss_hist, rewards_hist = [], []
    gamma = 0.9
    t_loss = 0

    with tf.Session() as sess:

        sess.run(init)

        # Check to see if model already exists. If it does, continue to train weights. If not, train new
        try:
            imported_meta = tf.train.import_meta_graph("../models/{}/{}.ckpt.meta".format(agent.model, agent.model))
            imported_meta.restore(sess, tf.train.latest_checkpoint('../models/{}/'.format(agent.model)))
            print("Retraining old model...")
        except OSError:
            print("Training new model...")

        for epoch in range(epochs):
            # eps_anneal = max(0.01, (epochs - 1.1 * epoch) / epochs)
            eps_anneal = 0.01

            boards_prev_log, actions_log, boards_log, rewards_log, target_q_log = [], [], [], [], []

            # Generate training data from games
            for i in range(iters):

                # Play games with the agent epsilon annealing from 1 to 0
                first = random.choice([0, 1])

                game_data = game.play(agent, player, sess, q_vals, inputs, agent_first=first, e=eps_anneal)

                boards_prev, actions, boards, rewards, winning_move = game_data
                boards_prev_log.extend(boards_prev)
                actions_log.extend(actions)
                boards_log.extend(boards)
                rewards_log.extend(rewards)
                w = winning_move

            # Q-Value updates for one game. Training step at each state-action
            for i in range(len(actions_log)):
                free = free_bin(boards_log[i])

                q_vals_next = sess.run(q_vals, feed_dict={inputs: [boards_log[i]]})
                free_q_vals = [i[1] for i in enumerate(q_vals_next[0]) if i[0] in free]

                if free_q_vals:  # Checking its non-empty

                    max_q_next = np.max(free_q_vals)
                else:
                    max_q_next = 0

                target_q = sess.run(q_vals, feed_dict={inputs: [boards_prev_log[i]]})
                update = rewards_log[i] + gamma * max_q_next  # from Bellman equation

                # change = (update - target_q[0][actions_log[i].index(1)])
                # for b in range(len(actions_log[i])):
                #
                #     if b not in free:
                #         target_q[0][b] = -1
                #     else:
                #         pass

                # target_q[0][actions_log[i].index(1)] += change  # one hot encoded move
                target_q[0][actions_log[i].index(1)] = update  # one hot encoded move

                # if rewards_log[i] == -1:
                #     if w:
                #         target_q[0][w] += 0.3
                #

                sess.run([train_step, q_vals], feed_dict={inputs: [boards_prev_log[i]], q_vals_target: target_q[0]})
                step =[0, 0, 0]
            # step = sess.run([train_step, q_vals, loss], feed_dict={inputs: boards_prev_log, q_vals_target: target_q_log})
            t_loss += step[2]

            if (epoch+1) % 250 == 0:

                print("\n")

                print(target_q[0])
                print("\nEpoch: {}".format(epoch+1))
                print("Agent wins+: {}".format(game.agent_wins - game.random_wins))
                print("Avg Loss: {}".format(round(t_loss, 3)))
                print("Epsilon: {}".format(eps_anneal))
                rewards_hist.append((game.agent_wins - game.random_wins)/500)
                t_loss_hist.append(t_loss/500)

                game.agent_wins = 0
                game.random_wins = 0
                t_loss = 0

        save_path = saver.save(sess, "../models/{}/{}.ckpt".format(agent.model, agent.model))
        print("Model saved in {}".format(save_path))
        #

        # epoch_ax = [i for i in range(0, 5000, 50)]
        # t_loss_hist[0] = t_loss_hist[1]
        # rewards_hist[0] = rewards_hist[1]

        # plt.plot(rewrads)
        #
        # with open("../data/{}.csv".format("2d-r-v-l-35000g"), mode="a") as datafile:
        #     data_write = csv.writer(datafile, delimiter=",")
        #     data_write.writerow(epoch_ax)
        #     data_write.writerow(rewards_hist)
        #     data_write.writerow(t_loss_hist)



if __name__ == "__main__":
    p11 = AgentRL(epsilon=None, model="3d-working-50k-winner", training=True)
    p12 = AgentRL(epsilon=0.1, model="3d-working-50k-winner", training=False)
    # p13 = AgentRL(epsilon=None, model="3d-layers-nodes-3x200s", training=True)
    # p1x = AgentRL(epsilon=0.2, model="test_bin", training=False)
    pt = AgentRLTable(0.01, 0.9, 0.9, model="qtable.pkl")

    p2 = RandomPlayer()
    pS = SmartPlayer()
    p3 = HumanPlayer()
    game = Game(p11, p2)

    trainRL(p11, p12, iters=1, epochs=5000, write=False, layers=2)
    # trainRL(p12, pS, iters=10, epochs=1500, write=False, layers=2)
    # trainRL(p13, pS, iters=10, epochs=1500, write=False, layers=3)
