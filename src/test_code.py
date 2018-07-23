import tensorflow as tf



def free_squares(board):
    free = [i for i in range(9) if board[i] == 0]

    return free

tf.reset_default_graph()
imported_meta = tf.train.import_meta_graph("/tmp/model_2d.ckpt.meta")




# input_positions_ = tf.placeholder(tf.float32, shape=[None, 9])
# labels_ = tf.placeholder(tf.float32, shape=[None, 9])
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()

with tf.Session() as sess:

    imported_meta.restore(sess, tf.train.latest_checkpoint('/tmp/'))
    board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    # weights = sess.run(weights)
    # logits = sess.run('logits:0')
    # print(logits)

    # b = sess.run('bias')
    # input_positions_ = tf.placeholder(tf.float32, shape=[None, 9])
    #
    weights = sess.run('logits:0', feed_dict={'x:0': [board]})[0]
    print(weights)
    d = dict(zip(range(0, len(weights)), weights))
    free = free_squares(board=board)

    d = [(a, d[a]) for a in free]

    d = sorted(d, key=lambda x: x[1], reverse=True)

    print(d)

    # print("Labels: {}".format(labels_))

