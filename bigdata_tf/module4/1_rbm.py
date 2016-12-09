import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import Image
from utils import tile_raster_images


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# Visible, input layer bias
vb = tf.placeholder("float", [784])

# Bias of the hidden layer
hb = tf.placeholder("float", [500])

# Weights
W = tf.placeholder("float", [784, 500])


# Forward pass
v0 = tf.placeholder("float", [None, 784])  # input

_h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb) # probabilities of hidden units

h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0)))) #sample_h_given_X


# Backward pass - reconstructing v1 from h0
_v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W))+ vb) # reconstructed 'input'

v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1)))) #sample_v_given_h


# Second forward pass
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)


# Calculating CD (Contrastive Divergence) and update weights and biases
alpha = 1.0
w_pos_grad = tf.matmul(tf.transpose(v0), h0)  # outer product of input and hidden layer
w_neg_grad = tf.matmul(tf.transpose(v1), h1)  # outer product of reconstruction and "second-epoch" hidden layer
CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])
update_w = W + alpha * CD  # new weights
update_vb = vb + alpha * tf.reduce_mean(v0 - v1, 0)  # new visible layer bias
update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)  # new hidden layer bias


# calculate error - difference between visible input layer and it's reconstruction
err = tf.reduce_mean(tf.square(v0 - v1))


# EXAMPLE OF ONE pass with one dataset
# Initializing session and variables
cur_w = np.zeros([784, 500], np.float32)
cur_vb = np.zeros([784], np.float32)
cur_hb = np.zeros([500], np.float32)
# # prv - previous
prv_w = np.zeros([784, 500], np.float32)
prv_vb = np.zeros([784], np.float32)
prv_hb = np.zeros([500], np.float32)
#
# with tf.Session() as sess:
#     init = tf.initialize_all_variables()
#     sess.run(init)
#
#     firstError = sess.run(err, feed_dict={v0: trX, W: prv_w, vb: prv_vb, hb: prv_hb})
#     print firstError


# Parameters
epochs = 5  # we will go through all dataset 5 times
batch_size = 100
weights = []
errors = []

with tf.Session() as sess:
    for epoch in range(epochs):
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
            batch = trX[start:end]
            # calculate new weights, passing current ones
            cur_w = sess.run(update_w, feed_dict={v0: trX, W: prv_w, vb: prv_vb, hb: prv_hb})
            cur_vb = sess.run(update_vb, feed_dict={v0: trX, W: prv_w, vb: prv_vb, hb: prv_hb})
            cur_hb = sess.run(update_hb, feed_dict={v0: trX, W: prv_w, vb: prv_vb, hb: prv_hb})
            # set current weights to previous
            prv_w = cur_w
            prv_vb = cur_vb
            prv_hb = cur_hb
            if start % 10000 == 0:
                print 'Iteration %d ' % start
                errors.append(sess.run(err, feed_dict={v0: trX, W: cur_w, vb: cur_vb, hb: cur_hb}))
                weights.append(cur_w)
        print 'Epoch: %d' % epoch, 'reconstruction error: %f' % errors[-1]


# Example of sampling
# with tf.Session() as sess:
#     a = tf.constant([0.7, 0.1, 0.8, 0.2])
#     print sess.run(a)
#     b = sess.run(tf.random_uniform(tf.shape(a)))
#     print b
#     print sess.run(a - b)
#     print sess.run(tf.sign(a - b))
#     print sess.run(tf.nn.relu(tf.sign(a - b)))
