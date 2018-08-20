import random
import numpy as np
import tensorflow as tf
from player import *
from ttt_3D import Game



def free_squares(board):
    free = [i for i in range(len(board)) if board[i] == 0]

    return free


def pass_net(shape=64, hidden_nodes=64):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    input_positions = tf.placeholder(tf.float32, shape=[1, shape], name="x")

    W_layer1 = weight_variable([shape, hidden_nodes])
    b_layer1 = bias_variable([hidden_nodes])

    W_layer2 = weight_variable([hidden_nodes, shape])
    b_layer2 = bias_variable([shape])

    h_layer1 = tf.matmul(input_positions, W_layer1) + b_layer1
    h_layer2 = tf.nn.relu(h_layer1)

    output = tf.matmul(h_layer2, W_layer2) + b_layer2

    y = tf.nn.softmax(output, name="q_values")

    pred = tf.argmax(y, 1)

    return input_positions, y, pred


def trainRL(agent, player, iters):
    tf.reset_default_graph()

    inputs, q_vals, pred = pass_net(64, 64)
    q_vals_target = tf.placeholder(tf.float32, shape=[1, 64])
    loss = tf.reduce_sum(tf.square(q_vals_target - q_vals))
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
    # update_model = train_step

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    t_loss = 0
    total_r = 0
    gamma = 0.9

    with tf.Session() as sess:

        sess.run(init)

        # Check to see if model already exists. If it does, continue to train weights
        try:
            imported_meta = tf.train.import_meta_graph("../models/{}/{}.ckpt.meta".format(agent.model, agent.model))
            imported_meta.restore(sess, tf.train.latest_checkpoint('../models/{}/'.format(agent.model)))
            print("Retraining old model...")
        except OSError:
            print("Training new model...")

        # Generate training data from games
        for i in range(iters):
            if i % 100 == 0:
                print("\nIteration {}".format(i))
                print("Agent wins+: {}".format(game.agent_wins - game.random_wins))
                game.agent_wins = 0
                game.random_wins = 0

            game_data = game.play(agent, player, sess, pred, q_vals, inputs, agent_first=i%2, e=(iters - i + 1) / iters)
            board_prevs, actions, boards, rewards = game_data

            for i in range(len(actions)):
                free = free_squares(boards[i])

                if i == len(actions) - 1:
                    r = rewards[i]

                else:
                    q_vals_next = sess.run([q_vals], feed_dict={inputs: [boards[i]]})
                    free_q_vals = [i[1] for i in enumerate(q_vals_next[0][0]) if i[0] in free]
                    max_q_next = np.max(free_q_vals)

                    r = gamma * max_q_next

                target_q = sess.run([q_vals], feed_dict={inputs: [board_prevs[i]]})
                target_q[0][0][actions[i]] = r

                sess.run([train_step, q_vals, loss],
                         feed_dict={inputs: [board_prevs[i]], q_vals_target: target_q[0]})

        save_path = saver.save(sess, "../models/{}/{}.ckpt".format(agent.model, agent.model))
        print("Model saved in {}".format(save_path))

        print("Agent wins: {}".format(game.agent_wins))
        print("Draws: {}".format(game.draws))
        print("Random wins: {}".format(game.random_wins))


p1 = AgentRL(epsilon=None, model="badline_3d", training=True)
# p1x = AgentRL(epsilon=0.1, model="rl_model_3d", training=True)
p2 = RandomPlayer()
game = Game(p1, p2)

trainRL(p1, p2, iters=2000)
