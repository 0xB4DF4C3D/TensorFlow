import tensorflow as tf
import numpy as np

X = tf.placeholder('float', [4,2])
Y = tf.placeholder('float', [4,1])

x = [[0,0],[0,1],[1,0],[1,1]]
y = [[0],[1],[1],[1]]

W = tf.Variable(tf.random_uniform([2,1], -1., 1.))
b = tf.constant([[-.5]]*4)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
#print(sess.run(tf.where(tf.reshape(tf.matmul(X,W)+b,[4]) > 0,[1,1,1,1],[0,0,0,0]),{X:x}))
print(sess.run(b))
hypothesis = tf.matmul(X,W)+b
test = tf.where(tf.reshape(tf.matmul(X,W)+b,[4]) > 0, [1.]*4, [0.]*4)
print(sess.run(hypothesis,{X:x}))
cost = tf.reduce_mean(tf.square(hypothesis - Y))
print('!',tf.trainable_variables())
print('cost',sess.run(cost,{X:x, Y:y}))
train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)


for step in range(101):
  sess.run(train, {X:x, Y:y})
  if step % 10 == 0:
    print(step, sess.run(cost, {X:x, Y:y}))
    print('\t', sess.run(hypothesis, {X:x}))
