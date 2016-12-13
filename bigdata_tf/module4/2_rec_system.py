#Tensorflow library. Used to implement machine learning models
import tensorflow as tf
#Numpy contains helpful functions for efficient mathematical calculations
import numpy as np
#Dataframe manipulation library
import pandas as pd
#Graph plotting library
# import matplotlib.pyplot as plt

movies_df = pd.read_csv('ml-1m/movies.dat', sep='::', header=None)
ratings_df = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None)

movies_df.columns = ['MovieID', 'Title', 'Genres']
movies_df['List Index'] = movies_df.index
ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

# Merging movies_df with ratings_df by MovieID
merged_df = movies_df.merge(ratings_df, on='MovieID')
merged_df = merged_df.drop('Timestamp', axis=1).drop('Title', axis=1).drop('Genres', axis=1)

userGroup = merged_df.groupby('UserID')
# print userGroup.head()

# Amount of users for training
amount_of_users = 1000
trX = []

# for each user in group
for userID, curUser in userGroup:
    # stores every movie rating
    temp = [0] * len(movies_df)
    # For each movie in curUser's movie list
    for num, movie in curUser.iterrows():
        # Divide the rating by 5 (to normalize)
        temp[movie['List Index']] = movie['Rating'] / 5.0
    trX.append(temp)
    if amount_of_users == 0: break
    amount_of_users-=1


# RBM Part

hidden_units = 20
visible_units = len(movies_df)

vb = tf.placeholder("float", [visible_units]) # bias from hidden layer to visible
hb = tf.placeholder("float", [hidden_units]) # bias from visible layer to hidden
W = tf.placeholder("float", [visible_units, hidden_units])

# Phase 1 - encoding input
v0 = tf.placeholder("float", [None, visible_units]) # input
_h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)  # probabilities of hidden units
h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))  #sample_h_given_X

# Phase 2 - reconstruction
_v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W))+ vb) # reconstructed 'input'
v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1)))) #sample_v_given_h
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

alpha = 1.0
w_pos_grad = tf.matmul(tf.transpose(v0), h0)
w_neg_grad = tf.matmul(tf.transpose(v1), h1)
CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])
update_w = W + alpha * CD
update_vb = vb + alpha * tf.reduce_mean(v0 - v1, 0)
update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

err = v0 - v1
err_sum = tf.reduce_mean(err * err)

cur_w = np.zeros([visible_units, hidden_units], np.float32)
cur_vb = np.zeros([visible_units], np.float32)
cur_hb = np.zeros([hidden_units], np.float32)

prv_w = np.zeros([visible_units, hidden_units], np.float32)
prv_vb = np.zeros([visible_units], np.float32)
prv_hb = np.zeros([hidden_units], np.float32)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

epochs = 15
batch_size = 100
errors = []
for i in range(epochs):
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        batch = trX[start:end]
        cur_w = sess.run(update_w, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_vb = sess.run(update_vb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_hb = sess.run(update_hb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        prv_w = cur_w
        prv_vb = cur_vb
        prv_hb = cur_hb
    errors.append(sess.run(err_sum, feed_dict={v0: trX, W: prv_w, vb: prv_vb, hb: prv_hb}))
    print errors


# input - movies in dataset, rating for one user, from 0 to 1
# 0 - means movie was not seen
# on reconstruction this value will not longer be 0, it will be the rating of the movie for the user


# Prediction for user 75
input_user = [trX[75]]

# Feeding in user and reconstructing input
hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
feed = sess.run(hh0, feed_dict={v0: input_user, W: prv_w, hb: prv_hb})
rec = sess.run(vv1, feed_dict={hh0: feed, W: prv_w, vb: prv_vb})

movies_df["Recommendation Score"] = rec[0]
print movies_df.sort(["Recommendation Score"], ascending=False).head(20)

