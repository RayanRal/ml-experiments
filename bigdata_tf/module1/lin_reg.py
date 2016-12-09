import tensorflow as tf
import numpy as np
# import matplotlib.patches as mpatches
# import matplotlib.pyplot as plt

# X = np.arange(0.0, 5.0, 0.1)
# print X

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 3 + 2
y_data = np.vectorize(lambda y: y + np.random.normal(loc=0.0, scale=0.1))(y_data)

# print zip(x_data, y_data)[0:5]

a = tf.Variable(1.0)
b = tf.Variable(0.2)
y = a * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

train_data = []
with tf.Session() as session:
    session.run(init)
    for step in range(100):
        evals = session.run([train, a, b])[1:]
        print evals
        train_data.append(evals)
