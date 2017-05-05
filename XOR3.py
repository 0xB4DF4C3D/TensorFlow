import tensorflow as tf
import numpy as np

xy = np.loadtxt('train_XOR.txt', unpack=True, dtype='float32')
x_data = xy[:-1]
y_data = xy[-1]
print(x_data)
W1_1 = tf.Variable(tf.random_uniform([2,2], -1., 1.))
b1_1 = tf.Variable(tf.random_uniform([2,4], -1., 1.))
W2_1 = tf.Variable(tf.random_uniform([1,2], -1., 1.))
b2_1 = tf.Variable(tf.random_uniform([1,4], -1., 1.))

K1 = tf.sigmoid(tf.matmul(W1_1,x_data)) + b1_1
K2 = tf.sigmoid(tf.matmul(W2_1,K1)) + b2_1
cost = tf.reduce_sum(tf.square(K2 - y_data))
train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

test = tf.where(K2 > .5,[[1]*4],[[1]*4]) 

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(201):
  sess.run(train)
  if step % 20 == 0:
    print(step, sess.run(cost))
    print('  ', np.round(sess.run(K2)))
