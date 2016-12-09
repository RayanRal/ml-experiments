import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import signal
import matplotlib.pyplot as plt

x = [6, 2]
h = [1, 2, 5, 4]

y = np.convolve(x, h, 'valid')
# print y

I = [[255,   7,  3],
    [212, 240,  4],
    [218, 216, 230]]

g = [[-1, 1]]

# print ('Without zero padding \n')
# print ('{0} \n'.format(sg.convolve(I, g, 'valid')))


inp = tf.Variable(tf.random_normal([1, 10, 10, 1]))
filt = tf.Variable(tf.random_normal([3, 3, 1, 1]))
op = tf.nn.conv2d(inp, filt, strides=[1,1,1,1], padding='SAME')
op2 = tf.nn.conv2d(inp, filt, strides=[1,1,1,1], padding='VALID')

init = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(init)

    # print("Input \n{0} \n".format(inp.eval()))
    # print("Filter/Kernel \n {0} \n".format(filt.eval()))
    # print("Result/Feature Map with valid positions \n")
    # result = session.run(op)
    # print(result)
    # print('\n')
    # print("Result/Feature Map with padding \n")
    # result2 = session.run(op2)
    # print(result2)


im = Image.open('bird.jpg')  # type here your image's name

# uses the ITU-R 601-2 Luma transform (there are several
# ways to convert an image to grey scale)

image_gr = im.convert("L")
print("\n Original type: %r \n\n" % image_gr)

# convert image to a matrix with values from 0 to 255 (uint8)
arr = np.asarray(image_gr)

kernel = np.array([[ 0, 1, 0],
                   [ 1,-4, 1],
                   [ 0, 1, 0]])
grad = signal.convolve2d(arr, kernel, mode='same', boundary='symm')

print('GRADIENT MAGNITUDE - Feature map')

fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad), cmap='gray')