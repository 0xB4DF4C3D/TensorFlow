import tensorflow as tf

_x = tf.placeholder(tf.float32, [None,2], 'x-input')
_y = tf.placeholder(tf.float32, [4,1], 'y-input')

W1 = tf.Variable(tf.random_uniform([2,2], -1, 1), 'Weight1')
W2 = tf.Variable(tf.random_uniform([2,1], -1, 1), 'Weight2')

b1 = tf.Variable(tf.zeros([2]), 'Bias1')
b2 = tf.Variable(tf.zeros([1]), 'Bias2')

a2 = tf.sigmoid(tf.matmul(_x, W1) + b1)
hypothesis =  tf.sigmoid(tf.matmul(a2, W2) + b2)


cost = -tf.reduce_mean(_y * tf.log(hypothesis) + (1 - _y) * tf.log(1. - hypothesis))

train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

XOR_X = [[0,0],[0,1],[1,0],[1,1]]
XOR_Y = [[0],[1],[1],[0]]

sess = tf.Session()
writer = tf.train.SummaryWriter('./log/xor')
sess.run(tf.global_variables_initializer())

for step in range(10001):
  sess.run(train, {_x:XOR_X, _y:XOR_Y})
  if step % 100 == 0:
    print(step, sess.run(cost, {_x:XOR_X, _y:XOR_Y}))

print('0 0 : ', sess.run(hypothesis, {_x:[[0,0]]}))
print('0 1 : ', sess.run(hypothesis, {_x:[[0,1]]}))
print('1 0 : ', sess.run(hypothesis, {_x:[[1,0]]}))
print('1 1 : ', sess.run(hypothesis, {_x:[[1,1]]}))

