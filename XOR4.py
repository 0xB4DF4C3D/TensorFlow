import tensorflow as tf
import numpy as np

xy = np.loadtxt('train_XOR2.txt', unpack=True, dtype='float32')
x = np.transpose(xy[:-2])
y = xy[-2:]

print(x, y)

W_1 = tf.Variable(tf.random_uniform([3,8], -1., 1.))
W_2 = tf.Variable(tf.random_uniform([10,1], -1., 1.))
b_1 = tf.Variable(tf.random_uniform([10], -1., 1.))
b_2 = tf.Variable(tf.random_uniform([1], -1., 1.))

K1 = tf.sigmoid(tf.add(tf.matmul(W_1, x), b_1))
K2 = tf.sigmoid(tf.add(tf.matmul(W_2, K1), b_2))
cost = tf.reduce_sum(tf.square(y - K2))
train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

test = tf.where(K2 > 0.5, [[1]*4], [[0]*4])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
  sess.run(train)
  if step % 1000 == 0:
    print(step, sess.run(cost), sess.run(test))

