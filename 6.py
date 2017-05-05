import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

sess = tf.Session()

print(sess.run([output], feed_dict={input1:[[2],[4]], input2:[[3, 5]]}))

sess.close()
