import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1., 1.))
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

h = tf.matmul(W, X)
hypothesis = tf.div(1., 1.+tf.exp(-h))

cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
  sess.run(train, {X:x_data, Y:y_data})
  if step % 100 == 0:
    print(step, sess.run(cost, {X:x_data, Y:y_data}), sess.run(W))

print('-=-=-=-=-=-=-=-=-=-=-=-=-=-')

print(sess.run(hypothesis, {X:[[1],[2],[2]]}))
print(sess.run(hypothesis, {X:[[1],[5],[5]]}))
print(sess.run(hypothesis, {X:[[1,1],[3,4],[3,4]]}))
