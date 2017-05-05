import tensorflow as tf
import numpy as np

X2 = tf.placeholder('float',[None,2])
Y = tf.placeholder('float',[None,1])

W = tf.Variable(tf.random_uniform([2,1], -1, 1))
b = tf.Variable(tf.random_uniform([1], -1, 1))

hypothesis = tf.nn.sigmoid(tf.matmul(X, W) + b)
cost = tf.reduce_mean(tf.square(hypothesis - Y))

train = tf.train.GradientDescentOptimizer(0.6).minimize(cost)

test = tf.round(hypothesis)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

x1 = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
x2 = [[1,0,0],[1,1,0],[1,1,0],[1,1,0],[1,0,1],[0,0,1],[1,1,1],[0,1,1]]
#y = np.transpose([[0,1,1,1]])
y = np.transpose([[]])

for step in range(201):
  sess.run(train, {X:x, Y:y})
  if step % 5 == 0 : 
    print(sess.run(cost, {X:x, Y:y}))    
    print('0 , 0 : ', sess.run(test, {X:[[0,0]]}))
    print('0 , 1 : ', sess.run(test, {X:[[0,1]]}))
    print('1 , 0 : ', sess.run(test, {X:[[1,0]]}))
    print('1 , 1 : ', sess.run(test, {X:[[1,1]]}))


