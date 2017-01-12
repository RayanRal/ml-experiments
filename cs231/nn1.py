import numpy as np
import math

class Neuron(object):

    def __init__(self):
        self.weights = None
        self.bias = None

    def forward(self, inputs):
        cell_body_sum = np.sum(inputs * self.weights) + self.bias
        firing_rate = 1.0 / (1.0 + math.exp(-cell_body_sum))  #sigmoid activation
        return firing_rate



f = lambda x: 1.0 / (1.0 + np.exp(-x))
x = np.random.rand(3, 1)

W1 = np.random.rand(4, 3)
W2 = np.random.rand(4, 4)
W3 = np.random.rand(1, 4)

b1 = np.random.rand(4, 1)
b2 = np.random.rand(4, 1)
b3 = np.random.rand(4, 1)

h1 = f(np.dot(W1, x) + b1)
h2 = f(np.dot(W2, h1) + b2)
out = np.dot(W3, h2) + b3
