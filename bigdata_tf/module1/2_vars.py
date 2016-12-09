import tensorflow as tf

scalar = tf.constant([2])
vector = tf.constant([5, 6, 2])
matrix = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
tensor = tf.constant([ [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                       [[4, 5, 6], [7, 8, 9], [10, 11, 12]]
                     ])

with tf.Session() as session:
    result = session.run(scalar)
    print result

    result = session.run(vector)
    print result

    result = session.run(matrix)
    print result

    result = session.run(tensor)
    print result


state = tf.Variable(0)

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init_op = tf.initialize_all_variables()

with tf.Session() as session:
    r1 = session.run(init_op)
    print "r1 %s" % r1
    r2 = session.run(state)
    print "r2 %s" % r2
    for _ in range(3):
        r3 = session.run(update)
        print "r3 %s" % r3
        r4 = session.run(state)
        print "r4 %s" % r4

print "###############"

a = tf.placeholder(tf.float32)
b = a*2
with tf.Session() as session:
    print(session.run(b, feed_dict={a: 3.5}))