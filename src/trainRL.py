import random
import numpy as np
import tensorflow as tf
from player import *
from ttt import Game


def free_squares(board):
    free = [i for i in range(9) if board[i] == 0]

    return free


def pass_net(shape=9, hidden_nodes=10):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
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


def trainRL(agent, player, e=0.1, iter=50000, epochs=2):
    tf.reset_default_graph()

    inputs, q_vals, pred = pass_net(9, 15)
    q_vals_target = tf.placeholder(tf.float32, shape=[1, 9])
    loss = tf.reduce_sum(tf.square(q_vals_target - q_vals))
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
    # update_model = train_step

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    t_loss = 0
    total_r = 0
    game_log = []
    gamma = 0.99

    with tf.Session() as sess:

        sess.run(init)

        # Generate training data from games
        for i in range(iter):
            if i % 100 == 0:
                print("\nIteration {}".format(i))
            iter_reward = 0
            n = random.random()
            game_data = game.play(agent, player, sess, pred, q_vals, inputs, agent_first=n, e=(iter-i+1)/iter)
            game_log.append(game_data)

            for epoch in range(epochs):
                # print("\nEpoch: {}".format(epoch))

                # random.shuffle(game_log)
                # for g in game_log:
                board_prevs, actions, boards, rewards = game_data
                #weighted_reward = [reward[-1] * 0.99 ** 1 / (i + 1) for i in range(len(reward)-1, -1, -1)]

                for i in range(len(actions)):
                    free = free_squares(boards[i])

                    if i == len(actions)-1:
                        r = rewards[i]
                        # print(r)

                    else:
                        q_vals_next = sess.run([q_vals], feed_dict={inputs: [boards[i]]})
                        free_q_vals = [i[1] for i in enumerate(q_vals_next[0][0]) if i[0] in free]
                        max_q_next = np.max(free_q_vals)

                        r = gamma * max_q_next


                    targetQ = sess.run([q_vals], feed_dict={inputs: [board_prevs[i]]})


                    targetQ[0][0][actions[i]] = r
                    # print("Updating reward: {}".format(r))

                    l = sess.run([train_step, q_vals, loss],
                                 feed_dict={inputs: [board_prevs[i]], q_vals_target: targetQ[0]})


                    t_loss += l[2]
                    total_r += r
                    iter_reward += r

            # print("Loss: {}".format(t_loss))
            # print("Total Reward: {}".format(total_r))
            # print("Iter reward: {}".format(iter_reward))



        save_path = saver.save(sess, "../models/{}/{}.ckpt".format("rl_model", "rl_model"))
        print("Model saved in {}".format(save_path))

        print("Agent wins: {}".format(game.agent_wins))
        print("Draws: {}".format(game.draws))
        print("Random wins: {}".format(game.random_wins))


p1 = AgentRL(0.01, 0.9, 0.9, model="rl_model")
p2 = RandomPlayer()
game = Game(p1, p2)

trainRL(p1, p2)
