import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

width = 28
height = 28
flat = width * height
class_output = 10

x = tf.placeholder(tf.float32, shape=[None, flat])
y_ = tf.placeholder(tf.float32, shape=[None, class_output])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Layer 1 - Convolutional
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

convolve1 = conv2d(x_image, W_conv1) + b_conv1

h_conv1 = tf.nn.relu(convolve1)

h_pool1 = max_pool_2x2(h_conv1)

layer1 = h_pool1

# Layer 2 - Convolutional
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])  # need 64 biases for 64 outputs

convolve2 = conv2d(layer1, W_conv2) + b_conv2

h_conv2 = tf.nn.relu(convolve2)

h_pool2 = max_pool_2x2(h_conv2)

layer2 = h_pool2


# Layer 3 - Fully-connected
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

layer2_matrix = tf.reshape(layer2, [-1, 7*7*64]) #flattening

matmul_fc1 = tf.matmul(layer2_matrix, W_fc1) + b_fc1

h_fc1 = tf.nn.relu(matmul_fc1)

layer3 = h_fc1

# dropout - currently not used
keep_prob = tf.placeholder(tf.float32)
layer3_drop = tf.nn.dropout(layer3, keep_prob)


# Layer 4 - SoftMax
W_fc2 = weight_variable([1024, 10]) #1024 neurons
b_fc2 = bias_variable([10])

matmul_fc2 = tf.matmul(layer3, W_fc2) + b_fc2

y_conv = tf.nn.softmax(matmul_fc2)

layer4 = y_conv  # in layer 4 we have softmax result, prediction, same as Y in previous example


#model training
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(layer4), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(layer4,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    for i in range(1500):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
            print("step %d, training accuracy %g" % (i, float(train_accuracy)))
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    # evaluating model on test set
    print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))