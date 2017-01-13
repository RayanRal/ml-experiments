import numpy as np
# import matplotlib.pyplot as plt

#           LINEAR CLASSIFIER
#Generating data
N = 100  # points per class
D = 2  # dimensionality of data
K = 3  # number of classes
step_size = 1e-0
reg = 1e-3  # regularization parameter
X = np.zeros((N*K, D))  # each row - single example
y = np.zeros(N*K, dtype='uint8')  # class labels
num_examples = X.shape[0]

for j in xrange(K):
    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0, 1, N)  # radius
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = j

#data visualization
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
# plt.show()


# initialize parameters
W = 0.01 * np.random.rand(D, K)
b = np.zeros((1, K))

for i in xrange(200):
    # compute class scores - forward pass
    scores = np.dot(X, W) + b
    # print scores.shape

    # COMPUTING LOSS
    #   get unnormalized probabilities
    exp_scores = np.exp(scores)
    #   normalize them for each example (row)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # now each row sums to one
    correct_logprobs = -np.log(probs[range(num_examples), y])
    #   compute loss
    data_loss = np.sum(correct_logprobs) / num_examples  # data loss
    reg_loss = 0.5*reg*np.sum(W*W)  # regularization loss
    loss = data_loss + reg_loss
    # if i % 10 == 0:
    #     print "iteration %d: loss %f" % (i, loss)

    # analytic gradient and backprop
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples

    dW = np.dot(X.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)
    dW += reg * W

    # parameter update
    W += - step_size * dW
    b += - step_size * db

# evaluate training set accuracy
scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print 'training accuracy: %.2f' % (np.mean(predicted_class == y))



#       NEURAL NETWORK with 1 hidden layer
h = 100  # neurons in hidden layer
W1 = 0.01 * np.random.rand(D, h)
b1 = np.zeros((1, h))
W2 = 0.01 * np.random.rand(h, K)
b2 = np.zeros((1, K))

for i in xrange(10000):
    # evaluate class scores, forward pass
    hidden_layer = np.maximum(0, np.dot(X, W1) + b1)  # maximum - ReLU activation
    scores = np.dot(hidden_layer, W2) + b2

    # COMPUTING LOSS
    #   get unnormalized probabilities
    exp_scores = np.exp(scores)
    #   normalize them for each example (row)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # now each row sums to one
    correct_logprobs = -np.log(probs[range(num_examples), y])
    #   compute loss
    data_loss = np.sum(correct_logprobs) / num_examples  # data loss
    reg_loss = 0.5 * reg * np.sum(W * W)  # regularization loss
    loss = data_loss + reg_loss
    if i % 1000 == 0:
        print "iteration %d: loss %f" % (i, loss)

    # analytic gradient and backprop
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples

    # backpropagation
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)

    dhidden = np.dot(dscores, W2.T)  # gradient by outputs, not by weights!
    # backprop the ReLU non-linearity
    dhidden[hidden_layer <= 0] = 0
    dW1 = np.dot(X.T, dhidden)
    db1 = np.sum(dhidden, axis=0, keepdims=True)

    # add regularization gradient contribution
    dW2 += reg * W2
    dW1 += reg * W1

    # parameter update
    W1 += -step_size * dW1
    b1 += -step_size * db1
    W2 += -step_size * dW2
    b2 += -step_size * db2


# evaluate training set accuracy
hidden_layer = np.maximum(0, np.dot(X, W1) + b1)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print 'training accuracy: %.2f' % (np.mean(predicted_class == y))
