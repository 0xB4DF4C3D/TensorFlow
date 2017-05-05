import tensorflow as tf

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

add = tf.add(X, Y)
mul = tf.mul(X, Y)

add_hist = tf.summary.scalar('add_scalar', add)
mul_hist = tf.summary.scalar('mul_scalar', mul)

merged = tf.summary.merge_all()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    writer = tf.summary.FileWriter('./board/sample_1', sess.graph)

    for step in range(100):
        summary = sess.run(merged, feed_dict={X: step * 1.0, Y: 2.0})
        writer.add_summary(summary, step)
        print(summary, step)
