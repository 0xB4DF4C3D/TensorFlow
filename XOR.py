import tensorflow as tf
import numpy as np

xy = np.loadtxt('train_XOR.txt', dtype='float32', unpack=True)
x_data = xy[:-1]
y_data = xy[-1]

print(x_data, y_data)
X = tf.placeholder('float', [2,None])
Y = tf.placeholder('float', [None])

W = tf.Variable(tf.random_uniform([1,2], -1., 1.))
b = tf.Variable(tf.random_uniform([1], -1., 1.))

hypothesis = tf.nn.softmax(tf.matmul(W, X) + b)
print(hypothesis.get_shape())
cost = -tf.reduce_sum(Y * tf.log(hypothesis))
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(10000):
  sess.run(train, {X:x_data, Y:y_data})
  if step%1000 == 0:print(step, sess.run(W,{X:x_data, Y:y_data}), sess.run(b), sess.run(cost,{X:x_data, Y:y_data}))
  #print('\t',sess.run(hypothesis,{X:x_data}))

print('0 0 : ', sess.run(hypothesis,{X:[[0],[0]]}))
print('0 1 : ', sess.run(hypothesis,{X:[[0],[1]]}))
print('1 0 : ', sess.run(hypothesis,{X:[[1],[0]]}))
print('1 1 : ', sess.run(hypothesis,{X:[[1],[1]]}))
