from player import *
from ttt import *
import csv
import random


def train_ai(infile="data-30000.csv", outfile="model", hidden_units=10, hidden_layers=1, epochs=20,
             batch_size=100, learning_rate=0.25):

    tf.reset_default_graph()

    with open("../data/" + infile, newline='') as csvfile:
        reader = csv.reader(csvfile)
        win_data = [i for i in reader if i[10] == '1']

    inputs, labels = [], []


    for row in win_data:
        inputs.append([int(d) for d in row[:9]])
        labels.append([int(d) for d in row[9][1:len(row[9])-1].split(',')])

    input_positions_ = tf.placeholder(tf.float32, shape=[None, 9], name="x")
    labels_ = tf.placeholder(tf.float32, shape=[None, 9])

    w1 = tf.Variable(tf.truncated_normal([9, hidden_units], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[hidden_units]))
    h1 = tf.tanh(tf.matmul(input_positions_, w1) + b1)

    w2 = tf.Variable(tf.truncated_normal([hidden_units, 9], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[9]))
    h2 = tf.matmul(h1, w2) + b2

    logits = tf.nn.softmax(h2, name="logits")

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits,
        labels=labels_))

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    with tf.Session() as sess:

        sess.run(init)

        n_batches = int(len(win_data)/batch_size)

        print("Data instances: {}".format(len(win_data)))

        for epoch in range(epochs):

            d = list(zip(inputs, labels))
            random.shuffle(d)
            inputs, labels = zip(*d)
            inputs, labels = np.array(inputs), np.array(labels)

            for batch in range(n_batches):

                start = batch * batch_size
                end = start + batch_size

                sess.run(train_step,
                         feed_dict={input_positions_: inputs[start:end], labels_: labels[start:end]})

            print("Cost: {}".format(sess.run(cross_entropy,
                                             feed_dict={input_positions_: inputs,
                                                        labels_: labels})))

        save_path = saver.save(sess, "../models/{}.ckpt".format(outfile))
        print("Model saved in path: %s" % save_path)


train_ai(infile="ai-r-30000.csv",
         outfile="model_20_50v2",
         hidden_units=20,
         epochs=50,
         batch_size=50,
         learning_rate=0.15)