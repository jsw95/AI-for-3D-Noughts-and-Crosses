from player import *
from ttt import *
import csv
import random
import numpy as np


def train_ai(
        infile="3d-t-r-gl<25.csv",
        outfile="3d",
        hidden_units=10,
        hidden_layers=5,
        epochs=10,
        batch_size=100,
        learning_rate=0.25,
        shape=64):

    tf.reset_default_graph()

    model_name = "{}-{}-{}-{}-{}-{}".format(
            outfile, hidden_units, hidden_layers, epochs, batch_size, learning_rate)

    with open("../data/" + infile, newline='') as csvfile:
        reader = csv.reader(csvfile)
        win_data = [i for i in reader if (i[-1] == '1') and (int(i[-2]) < 25)]

    inputs, labels = [], []
    for row in win_data:
        inputs.append([int(d) for d in row[:shape]])
        labels.append([int(d) for d in row[shape][1:len(row[shape]) - 1].split(',')])

    with open("../models/info/{}.txt".format(model_name), "a+") as info:
        info.write("Training Data: {}\n".format(infile))
        info.write("Instances: {}\n".format(len(inputs)))
        info.write("Hidden Units: {}\n".format(hidden_units))
        info.write("Hidden Layers: {}\n".format(hidden_layers))
        info.write("Epochs: {}\n".format(epochs))
        info.write("Batch Size: {}\n".format(batch_size))
        info.write("Learning Rate: {}\n\n".format(learning_rate))

    input_positions_ = tf.placeholder(tf.float32, shape=[None, shape], name="x")
    labels_ = tf.placeholder(tf.float32, shape=[None, shape])

    def pass_net(net_input):
        w1 = tf.Variable(tf.truncated_normal([shape, hidden_units], stddev=0.1))
        b1 = tf.Variable(tf.constant(0.1, shape=[hidden_units]))
        h1 = tf.tanh(tf.matmul(net_input, w1) + b1)

        w2 = tf.Variable(tf.truncated_normal([hidden_units, shape], stddev=0.1))
        b2 = tf.Variable(tf.constant(0.1, shape=[shape]))
        h2 = tf.matmul(h1, w2) + b2

        return h2

    output = pass_net(input_positions_)

    if hidden_layers > 1:
        for hidden_layer in range(hidden_layers - 1):
            output = pass_net(output)

    logits = tf.nn.softmax(output, name="logits")

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits,
        labels=labels_))

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    with tf.Session() as sess:

        sess.run(init)

        n_batches = int(len(win_data) / batch_size)

        print("Data instances: {}".format(len(win_data)))
        print("Training...")

        for epoch in range(epochs):

            d = list(zip(inputs, labels))
            random.shuffle(d)
            inputs, labels = zip(*d)
            inputs, labels = np.array(inputs), np.array(labels)

            for batch in range(n_batches):
                start = batch * batch_size
                end = start + batch_size

                sess.run(train_step,
                         feed_dict={input_positions_: inputs[start:end],
                                    labels_: labels[start:end]})

            cost = sess.run(cross_entropy, feed_dict={input_positions_: inputs,
                                                      labels_: labels})
            cost = round(float(cost), 3)
            print("Cost: {}".format(cost))

            with open("../models/info/{}.txt".format(model_name), "a+") as info:
                info.write("Cost: {}\n".format(cost))
                info.close()

        save_path = saver.save(sess, "../models/{}/{}.ckpt".format(model_name, model_name))

        print("Model saved in {}".format(save_path))


if __name__ == "__main__":
    train_ai(infile="3d-r-r-gl<25.csv",
             hidden_units=64,
             hidden_layers=1,
             epochs=10,
             batch_size=15,
             learning_rate=0.5)

    
