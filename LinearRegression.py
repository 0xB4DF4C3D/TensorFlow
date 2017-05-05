import tensorflow as tf
import numpy as np
import pylab as pl

x = np.sort(np.random.random(100))*100
y = np.sort(np.random.random(100))*50 + 10

W = tf.Variable(.5)
b = tf.Variable(np.random.randn())

hypothesis = W*x + b
cost = tf.reduce_sum(tf.square(y - hypothesis))
train = tf.train.GradientDescentOptimizer(1e-6*2).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

sess.run(train)
pl.plot(x,sess.run(W)*x+sess.run(b))

arr = y-(sess.run(W)*x+sess.run(b))
for i in range(1, len(arr)):
  arr[i] = arr[i-1]+abs(arr[i])
arr *= 0.1
pl.plot(x,arr,'bo')

for step in range(40001):
  sess.run(train)
  if step % 4000 == 0 : 
    print('cost', sess.run(cost))
 
pl.plot(x,sess.run(W)*x+sess.run(b))

pl.plot(x,y,'go')

arr = y-(sess.run(W)*x+sess.run(b))
for i in range(1,len(arr)):
  arr[i] = arr[i-1]+abs(arr[i])
arr *= 0.1
pl.plot(x,arr,'ro')

pl.xlim(-20,110)
pl.ylim(-20,110)
pl.show()
