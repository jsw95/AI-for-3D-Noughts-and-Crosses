from player import *
from ttt import *
import csv
import random


def free_squares(board):
    free = [i for i in range(9) if board[i] == 0]

    return free


with open("../data/data.csv", newline='') as csvfile:
    reader = csv.reader(csvfile)
    win_data = [i for i in reader if i[10] == '1']

inputs, labels = [], []

for row in win_data:
    inputs.append([int(d) for d in row[:9]])
    labels.append([int(d) for d in row[9][1:len(row[9])-1].split(',')])


input_positions_ = tf.placeholder(tf.float32, shape=[None, 9])
labels_ = tf.placeholder(tf.float32, shape=[None, 9])

weights = tf.Variable(tf.truncated_normal([9, 9], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[9]))
y = tf.matmul(input_positions_, weights) + bias

logits = tf.nn.softmax(y)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits,
                labels=labels_))

train_step = tf.train.GradientDescentOptimizer(0.9).minimize(cross_entropy)


with tf.Session() as sess:

    sess.run(init)

    epochs = 10
    batch_size = 100
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

            probs = sess.run(logits,
                             feed_dict={input_positions_: inputs[start:end]})[0]

            sess.run(train_step,
                     feed_dict={input_positions_: inputs[start:end], labels_: labels[start:end]})

        print("Cost: {}".format(sess.run(cross_entropy,
                                         feed_dict={input_positions_: inputs,
                                                    labels_: labels})))
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)


print("this is a github commit check")