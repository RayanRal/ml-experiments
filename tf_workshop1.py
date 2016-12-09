import tensorflow as tf
import numpy as np

a = tf.constant(5.0)
b = tf.constant(3.0)
c = a + b
print("before session")
# f = np.zeros()

print(c)
# saver = tf.train.Saver  # save to disk

with tf.Session() as sess:
    print("\nstarted session")
    print(sess.run(c))
    # saver.save(sess, "~/workshop1")


weights = tf.Variable(tf.random_normal([100, 150], stddev=0.5), name="weights")
w2 = tf.Variable(weights.initialized_value(), name="w2")
biases = tf.Variable(tf.zeros([150]), name="biases")

# Pin to GPU
#     with tf.device("/gpu:0"):

# good way to init variable (placeholder) - with feed_dict
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)
with tf.Session() as sess:
    print("\nstarted session 2")
    print(sess.run(output, feed_dict={input1:[6.0], input2:[3.0]}))
