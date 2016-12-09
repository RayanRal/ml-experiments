import tensorflow as tf
import pandas as pd
import numpy as np

#
# Tensor-flow gradient descent script
#

input_data = pd.read_csv('data/train.csv', header=0)

test_images = pd.read_csv('data/test.csv')

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch = input_data.sample(frac=0.2)
    batch_xs = batch.drop('label', axis=1).as_matrix() / 255.0
    # one-hot encoding labels
    batch_ys = pd.get_dummies(batch['label']).as_matrix()

    # learning
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    # evaluation
    train_eval = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    print train_eval

print '-----------------+++++++++++++++++'

predict = sess.run(y, feed_dict={x: test_images.as_matrix() / 255.0})
pred = [[i + 1, np.argmax(one_hot_list)] for i, one_hot_list in enumerate(predict)]
submission = pd.DataFrame(pred, columns=['ImageId', 'Label'])
submission.to_csv('submission_logreg.csv', index=False)

# predictions = model.predict(testDataForPrediction)
# classifications = []
# for i in chunker(test_images, 1):
#     classification = sess.run(tf.argmax(y, 1), feed_dict={x: i})
#     # print classification
#     classifications.append(classification)
#
# results = zip(test_images.index.values, classifications)
#
# print "Predictions {}".format(results)
#