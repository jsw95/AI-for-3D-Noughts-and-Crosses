import tensorflow as tf

tf.reset_default_graph()

input_positions_ = tf.placeholder(tf.float32, shape=[None, 9])
labels_ = tf.placeholder(tf.float32, shape=[None, 9])

saver = tf.train.Saver()

with tf.Session() as sess:

    saver.restore(sess, "/tmp/model.ckpt")

    print("Labels: {}".format(labels_))

