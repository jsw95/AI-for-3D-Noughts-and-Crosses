import tensorflow as tf


def free_squares(board):
    free = [i for i in range(9) if board[i] == 0]

    return free




tf.reset_default_graph()
imported_meta = tf.train.import_meta_graph("/tmp/model_2d_wl.ckpt.meta")


def move(board):

    with tf.Session() as sess:

        imported_meta.restore(sess, tf.train.latest_checkpoint('/tmp/'))

        weights = sess.run('logits:0', feed_dict={'x:0': [board]})[0]

        free = free_squares(board=board)

        d = [i for i in enumerate(weights) if i[0] in free]

        move_scores = sorted(d, key=lambda x: x[1], reverse=True)

        return move_scores[0][0]




move([0,0,0,0,0,0,0,0,0])  # correct 4
move([1,1,-1,1,-1,-1,0,0,0])  # correct 6
move([1,1,0,-1,-1,0,0,0,0])  # correct 2
move([-1,-1,0,1,1,0,0,0,0])  # correct 5
move([-1,-1,0,1,0,0,0,0,0])  # correct 5
