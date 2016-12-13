from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np


# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10


# Networks hyperparams
n_hidden_1 = 256 # features in first hidden layer
n_hidden_2 = 128 # features in second hidden layer
n_input = 784 # MNIST data, 28*28

X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input]))
}


# Building an encoder
def encoder(x):
    # encoder 1st layer with sigmoid activation function
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    # encoder 2nd layer with sigmoid activation function
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2


# Building a decoder
def decoder(x):
    # decoder 1st layer with sigmoid activation function
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    # decoder 2nd layer with sigmoid activation function
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op  # Prediction
y_true = X  # target are input data

# Loss and optimizer
cost = tf.reduce_mean(tf.pow((y_true - y_pred), 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(init)
    total_batch = int(mnist.train.num_examples / batch_size)

    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = session.run([optimizer, cost], feed_dict={X: batch_xs})
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))
    print("Optimization Finished!")


