import tensorflow as tf
import numpy as np


X_gen = np.arange(100, step=.1)
Y_gen = X_gen + 10 * np.cos(X_gen/5)

n_samples = 1000
batch_size = 150
steps_number = 1000

X_gen = np.reshape(X_gen, (n_samples, 1))  # transpond array
Y_gen = np.reshape(Y_gen, (n_samples, 1))

X = tf.placeholder(tf.float32, shape=(batch_size, 1))  # reserved values for tf, not filled yet
Y = tf.placeholder(tf.float32, shape=(batch_size, 1))

with tf.variable_scope("linear_regression"):
    # todo - check why here is "get_variable"
    k = tf.get_variable("weights", (1, 1), initializer=tf.random_normal_initializer())
    b = tf.get_variable("bias", (1, ), initializer=tf.constant_initializer(0.0))

y_predicted = tf.matmul(X, k) + b
loss = tf.reduce_sum((Y - y_predicted)**2)

opt_operation = tf.train.AdamOptimizer().minimize(loss)
display_step = 100

with tf.Session() as sess:
    print("\nstarted session")
    sess.run(tf.initialize_all_variables())
    for i in range(steps_number):
        # select random minibatch
        indices = np.random.choice(n_samples, batch_size)
        X_batch, Y_batch = X_gen[indices], Y_gen[indices]
        # Do gradient descent step
        sess.run([opt_operation, loss], feed_dict={X: X_batch, Y: Y_batch})  # fill X and Y placeholders
        if(i+1)%display_step == 0:
            c = sess.run(loss, feed_dict={X: X_batch, Y: Y_batch})
            print("epoch:", '%04d' % (i + 1), "cost=", "{:.9f}".format(c), "k=", sess.run(k), "b=", sess.run(b))
