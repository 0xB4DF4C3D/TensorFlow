import tensorflow as tf

a = tf.constant(1)
v = tf.Variable(0)
op = tf.assign(v,tf.add(v, a))

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for i in range(10):
  print(sess.run(op))

